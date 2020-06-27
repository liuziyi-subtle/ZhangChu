from torchvision import transforms
from seed import SEED


def get_dataset(cls, data, targets, data_idx, cutout_length=0):
    """ 对训练集和测试集分别作变换 """
    MEAN = []
    STD = []
    normalize = [
        transforms.ToTensor(),
        # transforms.Normalize(MEAN, STD)
    ]
    coutout = []
    if cutout_length > 0:
        coutout.append(Cutout(coutout_length))

    train_transform = transforms.Compose(transf, normalize, cutout)
    valid_transform = transforms.Compose(normalize)

    dataset_train = SEED(
        data, targets, data_idx, train=True, transform=train_transform)
    dataset_valid = SEED(
        data, targets, data_idx, train=False, transform=train_transform)

    return dataset_train, dataset_valid
