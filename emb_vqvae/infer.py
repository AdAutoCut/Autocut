# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Author     ：Jian Jia
# File       : infer.py
# Time       ：2024/10/31 12:05
"""

import argparse
import logging
import os
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from data.dataset_merchant import EmbeddingDataset
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
        "--config-file", type=str, default="model/rqvae_cosloss_rqvae_CBsize1024_CBdim64_CBhead8_lr_1.00e-03_2025-07-10-13:36:27/config.yaml",
    )
    parser.add_argument('--train', action="store_true")

    args = parser.parse_args()

    return args


def build_index(id_to_emb):
    ids = torch.tensor(list(id_to_emb.keys()))
    embs = torch.stack(list(id_to_emb.values()), dim=0)
    return ids, embs


def find_top_k_results(emb1, emb2, k=5):
    emb1_norm = F.normalize(emb1, p=2, dim=1)   # (nq, d)
    emb2_norm = F.normalize(emb2, p=2, dim=1)   # (nc, d)
    # 计算相似度矩阵 (nq, nc)
    sim_matrix = emb1_norm @ emb2_norm.t()
    # 对每个查询，取 top k
    topk_vals, topk_idx = torch.topk(
        sim_matrix, k, dim=1, largest=True, sorted=True)
    # N, K = topk_idx.shape
    # device = topk_idx.device
    # # create a (N,) tensor [0,1,2,...,N-1]
    # gt = torch.arange(N, device=device)
    # hits = topk_idx == gt.unsqueeze(1)        # shape (N, K), bool
    # cum_hits = hits.cumsum(dim=1).clamp_max(1).bool()  # still (N, K)
    # recall_at_j = cum_hits.float().mean(dim=0)  # (K,) float
    return topk_idx, topk_vals


def infer_fn(config, args):
    # local_rank = int(os.environ["LOCAL_RANK"])
    # args.local_device_rank = max(local_rank, 0)
    # torch.cuda.set_device(args.local_device_rank)
    # args.device = torch.device("cuda", args.local_device_rank)
    # device = args.device
    args.local_device_rank = 0
    device = torch.device('cpu')

    # dist.init_process_group(backend="nccl")
    # args.rank = dist.get_rank()
    # args.world_size = dist.get_world_size()
    args.rank = 0

    model_path = os.path.join(
        config.load.model_path,
        f'{config.load.epoch_id}.pth')

    assert os.path.exists(model_path), f"Model {model_path} does not exist"

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

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # model = DDP(model, device_ids=[args.local_device_rank], find_unused_parameters=False)

    state_dicts = torch.load(model_path, map_location="cpu")
    lacked_key, unexpected_key = model.load_state_dict(
        state_dicts, strict=False)
    if is_master(args):
        logging.info("lacked_key is ")
        logging.info(lacked_key)
        logging.info("unexpected_key is ")
        logging.info(unexpected_key)

    eval_dataset = EmbeddingDataset(
        item_lmdb_root=config.dataset.data_root,
        stage='eval',
        infer_file=config.dataset.infer_file,
        test_file=config.dataset.infer_file
    )

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=8192,  # 8192
        num_workers=8,
        drop_last=False,
        shuffle=False,
    )

    cudnn.benchmark = True
    cudnn.deterministic = False

    model.eval()

    wf = open(config.infer.infer_path, 'w')

    loss_list = []
    indices_list = []
    probs_list = []
    count_list = []
    cosine_sim_list = []
    topk_ids_list = []
    gt_ids_list = []
    # db_ids, db_embs = build_index(eval_dataset.data)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dataloader)):
            photo_id, emb = batch
            emb = emb.to(device)
            rec_emb, [indices, commit_loss] = model(emb)
            # codes = model.quantizer.get_codes_from_indices(indices)
            topk_idx, topk_vals = find_top_k_results(rec_emb, db_embs)
            topk_ids = db_ids[topk_idx]
            topk_ids_list.append(topk_ids)
            gt_ids_list.append(photo_id)
            loss_rec = get_reconstruction_loss(config, rec_emb, emb)

            loss_list.append(loss_rec.item())
            indices_list.append(indices)

            encode_onehot = F.one_hot(indices.permute(1, 0), config.model.codebook_size).type(
                rec_emb.dtype)  # [nhead, bt, ncode]
            # encode_onehot = encode_onehot.view(-1, codebook_size)
            probs_tmp = torch.mean(
                encode_onehot, dim=1).cpu()  # [num_head, ncode]
            probs_list.append(probs_tmp)

            count_tmp = torch.sum(encode_onehot, dim=1).cpu()  # [nhead, ncode]
            count_list.append(count_tmp)
            # 统计 codebook_count 不为0的个数

            # 🔥 计算 cosine similarity
            if emb.shape == rec_emb.shape:
                sim = F.cosine_similarity(emb, rec_emb, dim=-1)  # [B]
                cosine_sim_list.append(sim.cpu())
            else:
                print(
                    f"[Warning] Shape mismatch: emb {emb.shape}, rec_emb {rec_emb.shape}, skip cosine sim")

            # write photo id and indices to wf file
            for j in range(len(photo_id)):
                indices_str = ','.join(
                    [str(x + idx * config.model.codebook_size) for idx, x in enumerate(indices[j].tolist())])
                wf.write(f"{photo_id[j]}\t{indices_str}\n")
            wf.flush()
            # if i % 100 == 0:
            #     logging.info(f"Processing {i}  of {len(eval_dataloader)}")

    wf.close()

    # 🔥 拼接所有 cosine sim
    if len(cosine_sim_list) > 0:
        all_cosine_sim = torch.cat(cosine_sim_list, dim=0).numpy()
        print(f"[Cosine Similarity Stats]")
        print(f"Mean: {np.mean(all_cosine_sim):.4f}")
        print(f"Std:  {np.std(all_cosine_sim):.4f}")
        print(f"Min:  {np.min(all_cosine_sim):.4f}")
        print(f"Max:  {np.max(all_cosine_sim):.4f}")

        # 🔥 可选：绘制直方图
        plt.hist(all_cosine_sim, bins=50, color='skyblue', edgecolor='black')
        plt.title("Cosine Similarity Distribution")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.savefig("cosine_sim_hist.png")  # 或者 plt.show()
    else:
        print("[Warning] cosine_sim_list is empty. Nothing to analyze.")

    all_gt_ids = torch.cat(gt_ids_list, dim=0)
    all_topk_ids = torch.cat(topk_ids_list, dim=0)
    wrong_index = torch.where(all_topk_ids[:, 0] != all_gt_ids)[0]
    hits = all_topk_ids == all_gt_ids.unsqueeze(1)        # shape (N, K), bool
    cum_hits = hits.cumsum(dim=1).clamp_max(1).bool()  # still (N, K)
    recall_at_j = cum_hits.float().mean(dim=0)  # (K,) float
    wrong_cases_gt = all_gt_ids[wrong_index]
    wrong_cases_ans = all_topk_ids[wrong_index]
    print("######## Wrong Case ##########")
    for i in range(len(wrong_cases_gt)):
        print(wrong_cases_gt[i].item(), end=': ')
        print(wrong_cases_ans[i].tolist())
    print("##############################")
    # e.g. to print Recall@1, Recall@2, … Recall@K:
    for j, r in enumerate(recall_at_j, start=1):
        print(f"Recall@{j} = {r:.3f}")


    avg_probs = torch.stack(probs_list, dim=0).mean(0)  # [num_head, ncode]
    codebook_count = torch.stack(count_list, dim=0).sum(0)  # [num_head, ncode]
    utilization = torch.sum(codebook_count > 0, dim=1) / \
        config.model.codebook_size  # [num_head, ]
    perplexity = torch.exp(-torch.sum(avg_probs *
                           torch.log(avg_probs + 1e-10), dim=1))  # [num_head, ]

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
