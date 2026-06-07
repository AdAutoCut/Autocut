import argparse
import logging
import os
import sys
import time
import torch.nn.functional as F
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import wandb
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataset_merchant import prepare_data_loader, EmbeddingDataset
from modeling.vq_vae import RQVAE, MheadVQVAE, get_perplexity, get_utilization
from utils.losses import L1_loss, L2_loss, cos_loss, mixed_loss
from utils.scheduler import cosine_lr

# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def is_master(args):
    return args.rank == 0


def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf


# def get_num_params(param):
#     """
#     Returns the number of parameters in a given parameter tensor.
#     Handles specific cases like Params4bit and zero-initialized tensors.
#     """
#     num_params = param.numel()
#
#     # If using DS Zero 3 and the weights are initialized empty
#     if num_params == 0 and hasattr(param, "ds_numel"):
#         num_params = param.ds_numel
#
#     # Handle 4bit params case
#     if param.__class__.__name__ == "Params4bit":
#         num_params = num_params * 2
#
#     return num_params


def get_nb_trainable_parameters(model):
    """
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    trainable_percentage = 100 * trainable_params / all_params if all_params > 0 else 0

    logging.info(
        f"trainable params: {trainable_params:,d} || all params: {all_params:,d} || trainable%: {trainable_percentage:.2f}"
    )

    return trainable_params, all_params


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, default="configs/exp_init.yaml",
    )
    parser.add_argument('--train', action="store_true")

    args = parser.parse_args()

    return args


def test_fn(config, model, eval_dataloader, epoch, device, wb):
    model.eval()
    with torch.no_grad():
        all_cosine_sims = []
        for i, batch in enumerate(eval_dataloader):
            photo_id, emb = batch
            emb = emb.to(device)
            rec_emb, [indices, commit_loss] = model(emb)

            perplexity = get_perplexity(indices.permute(1, 0), config.model.codebook_size, rec_emb.dtype)
            utilization = get_utilization(indices.permute(1, 0), config.model.codebook_size, rec_emb.dtype)
            loss_rec = get_reconstruction_loss(config, rec_emb, emb)
            
            cosine_sim = F.cosine_similarity(emb, rec_emb, dim=-1)
            all_cosine_sims.append(cosine_sim.cpu())

        all_cosine_sims = torch.cat(all_cosine_sims, dim=0)
        avg_cosine_sim = all_cosine_sims.mean().item()

    if is_master(args):

        if wb != None:
            perplexity_dict = {}
            for i in range(config.model.num_head):
                perplexity_dict[f"TEST_perplexity_{i}"] = perplexity[i].item()

            utilization_dict = {}
            for i in range(config.model.num_head):
                utilization_dict[f"TEST_utilization_{i}"] = utilization[i].item()

            metric_dict = {
                "epoch": epoch + 1,
                "test loss": loss_rec.item(),
                "cosine_similarity": avg_cosine_sim,
            }
            wb.log({**metric_dict, **perplexity_dict, **utilization_dict})

        logging.info(
            f"Test set performance \n"
            f"Rec Loss {loss_rec.item():.4f}, "
            f"perplexity {perplexity}, "
            f"utilization {utilization}, "
            f"cosine sim {avg_cosine_sim:.4f}"
        )


def get_reconstruction_loss(config, rec_emb, emb):
    if config.trainer.loss_type == 'L1':
        loss_rec = L1_loss(rec_emb, emb)
    elif config.trainer.loss_type == 'L2':
        loss_rec = L2_loss(rec_emb, emb)
    elif config.trainer.loss_type == 'cos':
        loss_rec = cos_loss(rec_emb, emb)
    elif config.trainer.loss_type == 'mixed':
        loss_rec = mixed_loss(rec_emb, emb)
    else:
        raise ValueError(f"Unknown loss type: {config.trainer.loss_type}")
    return loss_rec


def train_fn(config, args):
    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_device_rank = max(local_rank, 0)
    # args.local_device_rank = 0
    torch.cuda.set_device(args.local_device_rank)
    args.device = torch.device("cuda", args.local_device_rank)
    # args.device = torch.device("cpu")
    device = args.device

    dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()
    # args.rank = 0

    # log the process of training
    current_time = datetime.now()
    time_str = current_time.strftime("%Y-%m-%d-%H:%M:%S")

    model_desc = (
        f"{config.experiment.name}_"
        f"{config.model.type}_"
        f"CBsize{config.model.codebook_size}_"
        f"CBdim{config.model.codebook_dim}_"
        f"CBhead{config.model.num_head}_"
        f"lr_{config.trainer.learning_rate:.2e}_"
        f"share_{config.model.shared_codebook}"
        f"{time_str}"
    )
    logging.info(f"model_desc: {model_desc}")

    save_dir = os.path.join(config.model.save_dir, model_desc)
    os.makedirs(save_dir, exist_ok=True)
    config.load.model_path = save_dir
    OmegaConf.save(config=config, f=os.path.join(save_dir, 'config.yaml'))

    if is_master(args) and config.wandb.is_use:
        wb = wandb.init(
            project=config.wandb.project,
            name=model_desc,
            config=args,
        )
    else:
        wb = None

    if config.model.type == "mhead":
        model = MheadVQVAE(
            input_dim=config.model.input_feature_size,
            codebook_dim=config.model.codebook_dim,
            codebook_size=config.model.codebook_size,
            num_head=config.model.num_head,
            encoder_mlp_size=config.model.encoder_mlp_size,
            decoder_mlp_size=config.model.decoder_mlp_size,
            shared_codebook=config.model.shared_codebook
        ).to(device)
    elif config.model.type == "rqvae":
        model = RQVAE(
            input_dim=config.model.input_feature_size,
            codebook_dim=config.model.codebook_dim,
            codebook_size=config.model.codebook_size,
            num_quantizers=config.model.num_head,
            encoder_mlp_size=config.model.encoder_mlp_size,
            decoder_mlp_size=config.model.decoder_mlp_size,
            shared_codebook=config.model.shared_codebook
        ).to(device)
    else:
        raise ValueError(f"Unknown model type {config.model.type}")

    # calculate the number of parameters of the model
    get_nb_trainable_parameters(model)

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = DDP(model, device_ids=[args.local_device_rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.trainer.learning_rate, betas=(0.9, 0.98), weight_decay=config.trainer.weight_decay
    )

    # train_dataset, eval_dataset = prepare_dataset(config.dataset.data_root)

    train_dataset = EmbeddingDataset(
        config.dataset.data_root,
        stage='train',
        train_file=config.dataset.train_file,
    )

    eval_dataset = EmbeddingDataset(
        config.dataset.data_root,
        stage='eval',
        test_file=config.dataset.test_file,
    )

    train_dataloader, eval_dataloader = prepare_data_loader(
        train_dataset, eval_dataset,
        config.dataset.batch_size,
        config.dataset.num_workers
    )

    total_steps = config.trainer.max_epoch * len(train_dataloader)
    warmup_steps = int(total_steps * config.trainer.warmup)
    scheduler = cosine_lr(
        optimizer, config.trainer.learning_rate, config.trainer.min_lr,
        warmup_steps, total_steps
    )

    cudnn.benchmark = True
    cudnn.deterministic = False

    cur_step = 0

    for epoch in range(config.trainer.max_epoch):

        if epoch > 0:
            # reset dataloader sampler
            train_dataloader, eval_dataloader = prepare_data_loader(
                train_dataset, eval_dataset,
                config.dataset.batch_size,
                config.dataset.num_workers,
                epoch_id=epoch
            )

        if args.local_device_rank == 0:
            logging.info(f"epoch {epoch}")

        model.train()

        data_start_time = time.time()
        for photo_id, emb in train_dataloader:
            # import pdb; pdb.set_trace()

            data_time = time.time() - data_start_time
            model_start_time = time.time()
            scheduler(cur_step)
            emb = emb.to(device)
            optimizer.zero_grad()
            rec_emb, [indices, commit_loss] = model(emb)

            perplexity = get_perplexity(indices.permute(1, 0), config.model.codebook_size, rec_emb.dtype)
            loss_rec = get_reconstruction_loss(config, rec_emb, emb)

            loss = config.trainer.recon_weight * loss_rec + config.trainer.commit_weight * commit_loss
            loss.backward()
            optimizer.step()
            model_time = time.time() - model_start_time

            if cur_step % config.trainer.plot_every == 0 and args.local_device_rank == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                logging.info(
                    f"Epoch [{epoch}/{config.trainer.max_epoch}], "
                    f"Step [{cur_step}/{total_steps}]"
                    f"L_rec: {loss_rec:.6f}, "
                    f"L_cmt: {commit_loss:.6f}, "
                    f"L_all: {loss:.6f}, "
                    f"lr: {cur_lr:.6f}, "
                    f"Data time {data_time:.2f} s, "
                    f"Model time {model_time:.2f} s, "
                )

                if wb != None:
                    perplexity_dict = {}
                    for i in range(config.model.num_head):
                        perplexity_dict[f"perplexity_{i}"] = perplexity[i].item()

                    metric_dict = {
                        "epoch": epoch + 1,
                        "step": cur_step + 1,
                        "learning rate": optimizer.param_groups[0]["lr"],
                        "loss": loss.item(),
                        "loss_rec": loss_rec.item(),
                        "commit_loss": commit_loss.item(),
                        "Data Time": data_time,
                        "Model Time": model_time,
                    }
                    wb.log({**metric_dict, **perplexity_dict})

            cur_step += 1
            data_start_time = time.time()

        if (epoch + 1) % config.trainer.save_every == 0 and args.local_device_rank == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'{epoch}.pth'))
        test_fn(config, model, eval_dataloader, epoch, device, wb)


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config_file)
    train_fn(config, args)
