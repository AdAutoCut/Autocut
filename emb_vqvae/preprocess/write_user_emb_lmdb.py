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

    root_dir = '/phd/content_ID/outer_rank/quantizer/mmu_embedding/user_emb/'
    nc_list = glob.glob(os.path.join(root_dir, 'llm_emb_nc/hive_files/part_*'))
    general_list = glob.glob(os.path.join(root_dir,  'llm_emb_general/hive_files/part_*'))
    jihuo_list = glob.glob(os.path.join(root_dir, 'llm_emb_jihuo/hive_files/part_*'))

    file_list = nc_list + general_list + jihuo_list

    print(f"there are total {len(file_list)} files")

    write_idx = 0
    for cur_file in tqdm(file_list):
        for line in tqdm(open(cur_file, 'r')):
            photo_id, emb_str = line.strip().split('\t')
            emb = np.array([float(value) for value in emb_str.split(',')])
            emb_n = emb / np.linalg.norm(emb)
            dump = pickle.dumps([photo_id, emb_n])
            write_txn.put(key=f"{write_idx}".encode('utf-8'), value=dump)
            write_idx += 1
            if write_idx % 50000 == 0:
                write_txn.commit()
                write_txn = write_env.begin(write=True)

    write_txn.put(key=b'num_samples', value=f"{write_idx}".encode('utf-8'))
    write_txn.commit()
    write_env.close()

    print(f"Finished serializing {write_idx} pairs into {write_lmdb_path}")
    print("done!")

    print(f"Finished serializing {write_idx} pairs into {args.output_path}")
    print("done!")

    # import numpy as np
    #
    # # 写入
    # fpath = np.memmap('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/recall_full/outer_full_revised.dat',
    #                   dtype='float32', mode='w+', shape=(idx2emb.shape[0], 512))
    # fpath[:] = idx2emb[:]
    # fpath.flush()
    # 读取
    # tmp_file = np.memmap('/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/recall_full/outer_full.dat',
    #                      dtype='float32', mode='r', shape=(26570729, 512))
