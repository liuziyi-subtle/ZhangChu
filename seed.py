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

        self.data = raw[[c for c in raw.columns if "band" in c]].values
        self.targets = raw["trans_type"].values  # subtype
        self.classes = np.unique(self.targets)
        self.class_to_idx = {_class: i for i,
                             _class in enumerate(self.classes)}

        if data_idx is None:
            train_idx, test_idx = train_test_split(
                self.targets, test_size=0.2, stratify=self.targets)
            data_idx = train_idx if self.train else test_idx

        self.data = self.data[data_idx, :].astype(np.float32)
        self.data = self.data.reshape(-1, 1, 200)
        self.data = self.data.transpose((0, 1, 2))  # convert to HWC

        self.targets = np.array([self.class_to_idx[x] for x in self.targets])
        self.targets = self.targets[data_idx]

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
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x_i, x_j = self.transform(x)

        # if self.target_transform is not None:
        #     target = self.target_transform(self.target)

        return x_i, x_j

    def __len__(self):
        return len(self.data)
