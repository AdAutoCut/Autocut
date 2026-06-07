import torch
import json
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import logging

import torch
import logging
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        item_lmdb_root: str,
        train_file: str = None,
        test_file: str = None,
        stage: str = 'train',
        infer_file=None,
    ) -> None:
        """
        Args:
            train_file: str train 阶段的 .pt 文件路径，格式为 {id: embedding}
            test_file: str test 阶段的 .pt 文件路径，格式为 {id: embedding}
            stage str: 'train' or 'eval'
        """

        super().__init__()
        self.stage = stage

        if stage == 'train':
            assert train_file is not None, "train_file must be provided for training stage."
            self.data = torch.load(train_file)
            logging.info(f"Loaded train data from {train_file}")

        elif stage == 'eval':
            assert test_file is not None, "test_file must be provided for eval stage."
            self.data = torch.load(test_file)
            logging.info(f"Loaded eval data from {test_file}")

        else:
            raise ValueError(f"Unsupported stage: {stage}")

        assert isinstance(self.data, dict), "Expected .pt file to contain a dict {id: embedding}"
        self.ids = list(self.data.keys())
        self.num_samples = len(self.ids)

        logging.info(f"{stage} Dataset contains {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        emb = self.data[id_]
        return int(id_), emb.float()



def prepare_data_loader(
        train_dataset, eval_dataset,
        batch_size, num_workers=8, epoch_id=0
):
    # train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=1234)
    # train_sampler.set_epoch(epoch_id)

    # eval_sampler = DistributedSampler(eval_dataset, shuffle=False)
    # eval_sampler.set_epoch(epoch_id)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        # sampler=train_sampler,
        drop_last=True
    )

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=128,  # eval 固定 batch_size 为大值
        num_workers=num_workers,
        # sampler=eval_sampler,
        drop_last=True
    )

    return train_dataloader, eval_dataloader
