from torchvision import transforms
from seed import SEED
import pandas as pd
from simclr.transformations import TransformsSimCLR


def get_dataset(path, name=None, pretrain=False):
    """ load dataset.
    """
    # TODO: select dataset according to name.
    transform = TransformsSimCLR()
    dataset_train = SEED(path, train=True, transform=transform)
    dataset_valid = SEED(path, train=False)

    if pretrain:
        return dataset_train

    return dataset_train, dataset_valid
