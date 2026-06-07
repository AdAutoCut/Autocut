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


if __name__ == '__main__':

    args = parse_args()

    # 获取父目录路径
    dir_name = os.path.dirname(args.output_path)
    if not os.path.exists(dir_name):
        print(f"Creating directory {dir_name}")
        os.makedirs(dir_name)

    # write pca embedding into lmdb

    if os.path.isdir(args.input_hive_path):
        file_list = glob.glob(os.path.join(args.input_hive_path, 'part_*'))
        file_list.sort()
        print(f"Found {len(file_list)} files in {args.input_hive_path}")
    else:
        file_list = [args.input_hive_path]

    write_idx = 0
    idx2photoid = {}
    idx2emb = []
    for cur_file in tqdm(file_list):
        for line in tqdm(open(cur_file, 'r')):
            photo_id, emb_str = line.strip().split('\x01')
            if emb_str is None or emb_str == '\\N':
                continue
            emb = np.array([float(value) for value in emb_str.split('\x02')])
            emb_n = emb / np.linalg.norm(emb)
            idx2photoid[write_idx] = photo_id
            idx2emb.append(emb_n)
            write_idx += 1

    idx2emb = np.stack(idx2emb, axis=0).astype(np.float32)
    print(f"idx2emb shape: {idx2emb.shape}")

    # with open(args.output_path, 'wb') as wf:
    #     pickle.dump([idx2photoid, idx2emb], wf)

    with open(os.path.join(args.output_path, 'emb_idx2photoid.pkl'),'wb') as wf:
        pickle.dump(idx2photoid, wf)




    # 写入
    # fpath = np.memmap(
    #     '/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/recall_full/outer_full_revised.dat',
    #     dtype='float32', mode='w+', shape=(idx2emb.shape[0], 512))
    dat_file = os.path.join(args.output_path, 'online_emb.dat')
    fpath = np.memmap(dat_file, dtype='float32', mode='w+', shape=(idx2emb.shape[0], 512))
    fpath[:] = idx2emb[:]
    fpath.flush()
    # 读取
    # tmp_file = np.memmap('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/recall_full/outer_full.dat',
    #                      dtype='float32', mode='r', shape=(26570729, 512))

    print(f"Finished serializing {write_idx} pairs into {args.output_path}")
    print("done!")
