import os
import numpy as np


class SEED():
    def __init__(self, data, targets, class_to_idx, data_idx, train=True, transform=None, target_transform=None):

        self.train = train
        self.data = []
        self.targets = []

        # TODO: 根据dataset的格式确定不同的组织方式, 最终汇聚成torch公用的格式. 目前, 是按照
        # df_objects的格式读取.
        self.data = data[data_idx, :]
        self.data = self.data.reshape(-1, 1, 1, -1)
        self.data = self.data.transpose((0, 1, -1, 1))  # convert to HWC
        self.targets = targets[data_idx]

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
