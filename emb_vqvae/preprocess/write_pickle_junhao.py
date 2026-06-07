# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Author     ：Jian Jia
# File       : wirte_pickle.py
# Time       ：2024/10/29 10:30
"""

# !/usr/bin/env python
# -*-coding:utf-8 -*-


import argparse
import glob
import os
import pickle
import random

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_hive_path", type=str, required=False,
        help="the directory which stores the image tsvfiles and the text jsonl annotations"
    )
    # parser.add_argument(
    #     "--split", type=str, required=True, help="specify the dataset split which this script processes"
    # )
    parser.add_argument(
        "--output_path", type=str, required=True, help="specify the directory which stores the output lmdb files. \
            If set to None, the lmdb_dir will be set to {args.data_dir}/lmdb"
    )
    return parser.parse_args()


def get_test_set():
    """
    获取测试集, 从全量集里随机抽取50w
    :return:
    """
    with open('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_1011-1018.pkl', 'rb') as rf:
        full_set = pickle.load(rf)

    embedding_size = full_set[1].shape[0]
    print(f" full_set shape: {embedding_size}")
    test_idx = random.sample(list(range(embedding_size)), k=500000)
    train_idx = list(set(range(embedding_size)) - set(test_idx))

    test_emb = full_set[1][test_idx]
    test_idx2photoid = {i: full_set[0][i] for i in test_idx}

    train_emb = full_set[1][train_idx]
    train_idx2photoid = {i: full_set[0][i] for i in train_idx}

    print(len(test_idx2photoid))
    assert test_emb.shape[0] == len(test_idx2photoid)

    with open('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_train_1011-1018.pkl', 'wb') as wf:
        pickle.dump([train_idx2photoid, train_emb], wf)

    with open('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_test_1011-1018.pkl', 'wb') as wf:
        pickle.dump([test_idx2photoid, test_emb], wf)


if __name__ == '__main__':

    args = parse_args()

    # 获取父目录路径
    dir_name = os.path.dirname(args.output_path)
    if not os.path.exists(dir_name):
        print(f"Creating directory {dir_name}")
        os.makedirs(dir_name)

    # write pca embedding into lmdb

    file_list = glob.glob(os.path.join(args.input_hive_path, 'part-*'))
    file_list.sort()

    write_idx = 0
    idx2photoid = {}
    idx2emb = []
    for cur_file in tqdm(file_list):
        for line in tqdm(open(cur_file, 'r')):
            photo_id, emb_str = line.strip().split('\x01')
            emb = np.array([float(value) for value in emb_str.split('\x02')])
            emb_n = emb / np.linalg.norm(emb)
            idx2photoid[write_idx] = photo_id
            idx2emb.append(emb_n)
            write_idx += 1

    idx2emb = np.stack(idx2emb, axis=0).astype(np.float32)
    print(f"idx2emb shape: {idx2emb.shape}")

    with open(args.output_path, 'wb') as wf:
        pickle.dump([idx2photoid, idx2emb], wf)

    print(f"Finished serializing {write_idx} pairs into {args.output_path}")
    print("done!")
