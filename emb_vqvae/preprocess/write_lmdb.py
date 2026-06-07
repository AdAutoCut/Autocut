# !/usr/bin/env python
# -*-coding:utf-8 -*-


import argparse
import glob
import os
import pickle

import lmdb
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
        "--lmdb_dir", type=str, required=True, help="specify the directory which stores the output lmdb files. \
            If set to None, the lmdb_dir will be set to {args.data_dir}/lmdb"
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    if not os.path.exists(args.lmdb_dir):
        print(f"Creating directory {args.lmdb_dir}")
        os.makedirs(args.lmdb_dir)

    # write pca embedding into lmdb
    write_lmdb_path = args.lmdb_dir
    write_env = lmdb.open(
        write_lmdb_path,
        map_size=1024 ** 4 * 5
    )
    write_txn = write_env.begin(write=True)

    if os.path.isdir(args.input_hive_path):
        file_list = glob.glob(os.path.join(args.input_hive_path, 'part_*'))
        file_list.sort()
        print(f"Found {len(file_list)} files in {args.input_hive_path}")
    else:
        file_list = [args.input_hive_path]

    write_idx = 0
    for cur_file in tqdm(file_list):
        for line in tqdm(open(args.input_hive_path, 'r')):
            photo_id, emb_str = line.strip().split('\x01')
            emb = np.array([float(value) for value in emb_str.split('\x02')])
            emb_n = emb / np.linalg.norm(emb)
            dump = pickle.dumps([photo_id, emb_n])
            write_txn.put(key=f"{write_idx}".encode('utf-8'), value=dump)
            write_idx += 1
            if write_idx % 5000 == 0:
                write_txn.commit()
                write_txn = write_env.begin(write=True)

    write_txn.put(key=b'num_samples', value=f"{write_idx}".encode('utf-8'))
    write_txn.commit()
    write_env.close()

    print(f"Finished serializing {write_idx} pairs into {write_lmdb_path}")
    print("done!")
