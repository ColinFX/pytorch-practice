import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import utils


transformer = ToTensor()


def fetch_dataset(type: str, data_dir: str) -> Dataset:
    """
    MNIST is a torchvision built-in datasets containing already splited `train-images-idx3-ubyte` and 
    `t10k-images-idx3-ubyte`, which can be automatically loaded as two Dataset instance. Each handwritten image in the 
    dataset is grayscale and of resolution 28*28, and shall be classified into 10 classes representing different digits. 

    Args:
        * type: (str) whether "train" or "test"
        * data_dir: (str) containing `MNIST/raw//train-images-idx3-ubyte` and `MNIST/raw/t10k-images-idx3-ubyte`
    
    Returns:
        * dataset: (Dataset) of the specific type
    """

    if type == "train":
        return datasets.MNIST(root=data_dir, train=True, transform=transformer, download=True)
    elif type == "test":
        return datasets.MNIST(root=data_dir, train=False, transform=transformer, download=True)
    else:
        raise KeyError("Incorrect type keyword {} not in [\"train\", \"test\"].".format(type))


def fetch_dataloader(type: str, data_dir: str, params: utils.Params) -> DataLoader:
    """
    Args: 
        * type: (str) whether "train" or "test"
        * data_dir: (str) containing `MNIST/raw//train-images-idx3-ubyte` and `MNIST/raw/t10k-images-idx3-ubyte`
        * params: (Params) hyperparameters

    Returns:
        * dataloader: (DataLoader) of the specified type
    """

    dataset = fetch_dataset(type, data_dir)
    return DataLoader(dataset, 
                      batch_size=params.batch_size, 
                      shuffle=True, 
                      num_workers=1)
