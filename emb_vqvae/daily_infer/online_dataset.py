# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Author     ：Jian Jia
# File       : online_dataset.py
# Time       ：2024/11/21 16:35
"""
import logging
import torch
import lmdb
import pickle
import os
import numpy as np


# class EmbeddingDataset_lmdb(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         item_lmdb_root: str,
#         stage: str = 'train'
#     ) -> None:
#         super().__init__()
#         self.item_lmdb_root = item_lmdb_root
#         self.stage = stage
#         if stage == 'train':
#             # 从训练集 LMDB 中读取数据
#             self.lmdb_env = lmdb.open(
#                 os.path.join(self.item_lmdb_root, 'train_set_lmdb'),
#                 readonly=True,
#                 lock=False,
#                 create=False,
#                 readahead=False,
#                 meminit=False
#             )
#             # 创建事务
#             self.lmdb_txn = self.lmdb_env.begin(buffers=True)
#             self.number_samples = int(
#                 self.lmdb_txn.get(key=b"num_samples").tobytes().decode("utf-8")
#             )
#             logging.info(f"Train LMDB file contains {self.number_samples} pairs.")
#
#         elif stage == 'eval':
#             self.lmdb_env = lmdb.open(
#                 os.path.join(self.item_lmdb_root, 'test_set_lmdb'),
#                 readonly=True,
#                 lock=False,
#                 create=False,
#                 readahead=False,
#                 meminit=False
#             )
#             # 创建事务
#             self.lmdb_txn = self.lmdb_env.begin(buffers=True)
#             self.number_samples = int(
#                 self.lmdb_txn.get(key=b"num_samples").tobytes().decode("utf-8")
#             )
#             logging.info(f"Test LMDB file contains {self.number_samples} pairs.")
#         else:
#             raise ValueError(f"Invalid stage: {stage}")
#
#         self.num_samples = self.number_samples
#
#         # print(self.stage, len(self.item_keys))
#
#     def __len__(self) -> int:
#         return self.num_samples
#
#     def __del__(self):
#         if hasattr(self, "lmdb_env"):
#             self.lmdb_env.close()
#
#     def __getitem__(self, idx):
#         # 从 lmdb 中根据key 读取数据
#         # 读取数据时，需要将 key 转换为 bytes 类型
#         photo_id, photo_emb = pickle.loads(self.lmdb_txn.get(key=str(idx).encode('utf-8')))
#         return photo_id, photo_emb.astype(np.float32)


class OnlineDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            idx2photo_list,
            photo_emb_array,
    ) -> None:
        super().__init__()
        # self.item_lmdb_root = item_lmdb_root

        assert len(idx2photo_list) == len(photo_emb_array)
        self.idx2photoid = idx2photo_list
        self.emb_array = photo_emb_array
        self.idx_list = list(idx2photo_list.keys())
        self.num_samples = len(self.idx_list)

    def __len__(self) -> int:
        return self.num_samples

    def __del__(self):
        if hasattr(self, "lmdb_env"):
            self.lmdb_env.close()

    def __getitem__(self, idx):
        # emb_array 是一个 tensor，其中的idx 是 从 0 到 len(self.idx_list)
        photo_emb = self.emb_array[idx]

        idx = self.idx_list[idx]
        # idx2photoid 是一个 dict，其中的key 不是从 0 到 len(self.idx_list)，而是经过采样得到的
        photo_id = self.idx2photoid[idx]

        return int(photo_id), photo_emb.astype(np.float32)
