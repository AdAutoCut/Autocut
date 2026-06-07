# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Author     ：Jian Jia
# File       : online_infer.py
# Time       ：2024/11/21 14:41
"""

import argparse
import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from daily_infer.online_dataset import OnlineDataset
from modeling.vq_vae import RQVAE, MheadVQVAE
from train import get_reconstruction_loss

# from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def is_master(args):
    return args.rank == 0


def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=str,  required=True,
    )
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--infer-file', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)
    parser.add_argument('--data-type', type=str, required=True, choices=['1', '2'])

    args = parser.parse_args()

    return args


def hive_file_process(input_file, data_type):
    emb_list = []
    idx2photo = {}
    index = 0
    for line in open(input_file, 'r'):
        try:
            if data_type == '1':
                photo_id, emb_str = line.strip().split('\t')
                emb = np.array([float(i) for i in emb_str.split(',')]).astype(np.float32)
            elif data_type == '2':
                photo_id, emb_str = line.strip().split('\x01')
                emb = np.array([float(i) for i in emb_str.split('\x02')]).astype(np.float32)
            assert len(emb) == 512
            idx2photo[index] = photo_id
            emb_list.append(emb)
            index += 1
        except Exception as e:
            print(f"{e} in line {line}")

    emb_list = np.array(emb_list)

    print(f"Processing {input_file}, shape: {emb_list.shape}")

    return idx2photo, emb_list


def infer_fn(config, args):
    local_rank = int(os.environ["LOCAL_RANK"])
    args.local_device_rank = max(local_rank, 0)
    torch.cuda.set_device(args.local_device_rank)
    args.device = torch.device("cuda", args.local_device_rank)
    device = args.device

    dist.init_process_group(backend="nccl")
    args.rank = dist.get_rank()
    args.world_size = dist.get_world_size()

    model_path = config.load.model_path

    assert os.path.exists(model_path), f"Model {model_path} does not exist"

    if config.model.type == "mhead":
        model = MheadVQVAE(
            input_dim=config.model.input_feature_size,
            codebook_dim=config.model.codebook_dim,
            codebook_size=config.model.codebook_size,
            num_head=config.model.num_head,
            encoder_mlp_size=config.model.encoder_mlp_size,
            decoder_mlp_size=config.model.decoder_mlp_size
        ).to(device)
    elif config.model.type == "rqvae":
        model = RQVAE(
            input_dim=config.model.input_feature_size,
            codebook_dim=config.model.codebook_dim,
            codebook_size=config.model.codebook_size,
            num_quantizers=config.model.num_head,
            encoder_mlp_size=config.model.encoder_mlp_size,
            decoder_mlp_size=config.model.decoder_mlp_size
        ).to(device)
    else:
        raise ValueError(f"Unknown model type {config.model.type}")

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[args.local_device_rank], find_unused_parameters=False)

    state_dicts = torch.load(model_path, map_location="cpu")
    lacked_key, unexpected_key = model.load_state_dict(state_dicts, strict=False)
    if is_master(args):
        logging.info("lacked_key is ")
        logging.info(lacked_key)
        logging.info("unexpected_key is ")
        logging.info(unexpected_key)

    idx2photo_list, photo_emb_array = hive_file_process(
        input_file=args.infer_file,
        data_type=str(args.data_type),
    )

    eval_dataset = OnlineDataset(
        idx2photo_list,
        photo_emb_array,
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=8192,
        num_workers=8,
        drop_last=False,
        shuffle=False,
    )

    cudnn.benchmark = True
    cudnn.deterministic = False

    model.eval()

    wf = open(args.output_file, 'w')

    loss_list = []
    indices_list = []
    probs_list = []
    count_list = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader)):
            photo_id, emb = batch
            emb = emb.to(device)
            rec_emb, [indices, commit_loss] = model(emb)

            loss_rec = get_reconstruction_loss(config, rec_emb, emb)

            loss_list.append(loss_rec.item())
            indices_list.append(indices)

            encode_onehot = F.one_hot(indices.permute(1, 0), config.model.codebook_size).type(
                rec_emb.dtype)  # [nhead, bt, ncode]
            # encode_onehot = encode_onehot.view(-1, codebook_size)
            probs_tmp = torch.mean(encode_onehot, dim=1).cpu()  # [num_head, ncode]
            probs_list.append(probs_tmp)

            count_tmp = torch.sum(encode_onehot, dim=1).cpu()  # [nhead, ncode]
            count_list.append(count_tmp)
            # 统计 codebook_count 不为0的个数

            # write photo id and indices to wf file
            for j in range(len(photo_id)):
                indices_str = ','.join(
                    [str(x + idx * config.model.codebook_size) for idx, x in enumerate(indices[j].tolist())])
                wf.write(f"{photo_id[j]}\t{indices_str}\n")
            wf.flush()
            # if i % 100 == 0:
            #     logging.info(f"Processing {i}  of {len(eval_dataloader)}")
    wf.flush()
    wf.close()

    avg_probs = torch.stack(probs_list, dim=0).mean(0)  # [num_head, ncode]
    codebook_count = torch.stack(count_list, dim=0).sum(0)  # [num_head, ncode]
    utilization = torch.sum(codebook_count > 0, dim=1) / config.model.codebook_size  # [num_head, ]

    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=1))  # [num_head, ]

    loss = sum(loss_list) / len(loss_list)

    logging.info(
        f"Test set performance \n"
        f"Rec Loss {loss:.6f}, "
        f"perplexity {perplexity}, "
        f"utilization {utilization}, "
    )


if __name__ == '__main__':
    args = parse_args()
    config = get_config(args.config_file)
    infer_fn(config, args)
