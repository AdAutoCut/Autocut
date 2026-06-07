import glob
import logging
import torch
import lmdb
import pickle
import os
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


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

def hive_file_process(file_list, data_type):
    emb_list = []
    idx2photo = {}
    index = 0
    for file in tqdm(file_list):
        for line in open(file, 'r'):
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


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            item_lmdb_root: str,
            stage: str = 'train',
            train_file=None,
            test_file=None,
            infer_file=None,
    ) -> None:
        super().__init__()
        # self.item_lmdb_root = item_lmdb_root
        self.train_file = train_file
        self.test_file = test_file
        self.infer_file = infer_file
        self.stage = stage
        if stage == 'train':
            logging.info(f"loading {stage} data from {self.train_file}")
            with open(self.train_file, 'rb') as rf:
                idx2photoid = pickle.load(rf)

                # 对于全量数据，使用 pickle 直接load 会 OOM.
                # del emb_array
                emb_array = np.memmap(
                    '/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/recall_full/outer_full_revised.dat',
                    dtype='float32', mode='r', shape=(26378119, 512))

        elif stage == 'eval':
            logging.info(f"loading {stage} data from {self.test_file}")
            with open(self.test_file, 'rb') as rf:
                idx2photoid, emb_array = pickle.load(rf)

        elif stage == 'infer':
            logging.info(f"loading {stage} data from {self.infer_file}")

            with open(self.infer_file, 'rb') as rf:
                idx2photoid = pickle.load(rf)

                # 对于全量数据，使用 pickle 直接load 会 OOM.
                # del emb_array
                # emb_array = np.memmap(
                #     '/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_training_1011_1018_use_seq_20241204/emb.dat',
                #     dtype='float32', mode='r', shape=(23083635, 512))
                # emb_array = np.memmap(
                #     '/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_training_250101/online_emb_250101_part1.dat',
                #     dtype='float32', mode='r', shape=(27391670, 512))
                # emb_array = np.memmap(
                #     '/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_training_1011_1018_use_seq_20241204/emb.dat',
                #     dtype='float32', mode='r', shape=(23083635, 512))
                emb_array = np.memmap(
                    '/phd/content_ID/outer_rank/quantizer/mmu_embedding/ad_outer/online_training_250217/online_emb.dat',
                    dtype='float32', mode='r', shape=(14952743, 512))

        else:
            raise ValueError(f"Invalid stage: {stage}")

        assert len(idx2photoid) == len(emb_array)
        self.idx2photoid = idx2photoid
        self.emb_array = emb_array
        self.idx_list = list(idx2photoid.keys())
        self.num_samples = len(self.idx_list)

        logging.info(f"{stage} Dataset contains {self.num_samples} pairs.")

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


def prepare_data_loader(
        train_dataset, eval_dataset,
        batch_size, num_workers=8, epoch_id=0
):
    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=1234)
    train_sampler.set_epoch(epoch_id)

    eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    eval_sampler.set_epoch(epoch_id)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        drop_last=True,

    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=8192,
        num_workers=num_workers,
        sampler=eval_sampler,
        drop_last=True,
    )

    return train_dataloader, eval_dataloader
