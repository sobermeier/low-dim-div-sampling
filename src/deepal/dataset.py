import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

from src.deepal.settings import DATA_ROOT

DATASET_PATH = DATA_ROOT + '/data'


# training and data settings for dataset
def get_transform(name):
    if name == 'MNIST':
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        raise NotImplementedError()


def get_dataset(name, path=DATASET_PATH):
    if name == 'MNIST':
        return get_MNIST(path)
    else:
        raise NotImplementedError()


def get_MNIST(path):
    raw_tr = datasets.MNIST(path, train=True, download=True)
    raw_te = datasets.MNIST(path, train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets

    class_index, class_count = np.unique(Y_tr, return_counts=True)
    N = np.zeros(10)
    N[class_index.astype(int)] = class_count
    N = torch.tensor(N)
    return X_tr, Y_tr, X_te, Y_te, N


def get_handler(name):
    if name == 'MNIST':
        return MNISTHandler
    else:
        raise NotImplementedError()


class MNISTHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x.numpy())
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)
