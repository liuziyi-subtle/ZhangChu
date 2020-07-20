import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class SEED():
    # train_list = list(range(500))
    # test_list = list(range(500, 600))

    def __init__(self, path, data_idx=None, train=True, transform=None, target_transform=None):
        self.train = train

        raw = pd.read_csv(path)
        # TODO: 根据dataset的格式确定不同的组织方式, 最终汇聚成torch公用的格式.
        # 目前, 是按照df_objects的格式读取.
        self.data = raw[[c for c in raw.columns if 'band' in c]].values
        self.targets = raw['subtype'].values
        self.classes = np.unique(self.targets)
        self.class_to_idx = {_class: i for i,
                             _class in enumerate(self.classes)}

        self.targets = [self.class_to_idx[x] for x in self.targets]

        if data_idx is None:
            # data_idx = self.train_list if self.train else self.test_list
            train_idx, test_idx = train_test_split(
                self.targets, test_size=0.2, random_state=0)
            data_idx = train_idx if self.train else test_idx

        self.data = self.data[data_idx, :].astype(np.float32)
        self.data = self.data.reshape(-1, 1, 200)
        self.data = self.data.transpose((0, 1, 2))  # convert to HWC

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
