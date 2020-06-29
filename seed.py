import os
import numpy as np
import pandas as pd


class SEED():
    train_list = list(range(500))
    test_list = list(range(500, 600))

    def __init__(self, path, data_idx=None, train=True, transform=None, target_transform=None):
        self.train = train

        raw = pd.read_csv(path)
        # TODO: 根据dataset的格式确定不同的组织方式, 最终汇聚成torch公用的格式.
        # 目前, 是按照df_objects的格式读取.
        self.data = raw[[c for c in raw.columns if 'band' in c]].values
        if data_idx is None:
            data_idx = self.train_list if self.train else self.test_list
        self.data = self.data[data_idx, :]
        self.data = self.data.reshape(-1, 1, self.data.shape[1], 1)
        self.data = self.data.transpose((0, 1, 2, 3))  # convert to HWC
        self.targets = raw['subtype'].values

        self.transform = transform
        self.target_transform = target_transform

    def _load_meta(self):
        pass

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        data, target = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(self.target)

        return data, target

    def __len__(self):
        return len(self.data)
