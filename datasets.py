from torchvision import transforms
from seed import SEED
import pandas as pd


def get_dataset(path):
    """
    1. 加载数据集为ndarray格式
    2. 添加transforms
    """
    # MEAN = []
    # STD = []
    # normalize = [
    #     transforms.ToTensor(),
    #     # transforms.Normalize(MEAN, STD)
    # ]
    # coutout = []
    # if cutout_length > 0:
    #     coutout.append(Cutout(coutout_length))

    # train_transform = transforms.Compose(transf, normalize, cutout)
    # valid_transform = transforms.Compose(normalize)

    dataset_train = SEED(path, train=True)
    dataset_valid = SEED(path, train=False)

    return dataset_train, dataset_valid
