"""specify how the data should be fed to the network"""

import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import utils


# preprocessing transformer
train_transformer = transforms.Compose([transforms.Resize(64),
                                        transforms.RandomHorizontalFlip(),      # extra random horizontal flip for train
                                        transforms.ToTensor()])
eval_transformer = transforms.Compose([transforms.Resize(64),
                                       transforms.ToTensor()])


class DatasetNameDataset(Dataset):
    """
    The torch Dataset class is an abstract class representing the dataset in map-style. The main task of a Dataset is 
    to return a pair of [input, label] every time it is called. Data preprocessing shall also be done in this class. A 
    custom Dataset should contain at least `__len__()` function to return the length of the dataset and `__getitem__()` 
    function to return one training example. 
    """

    def __init__(self, sub_data_dir: str, transform: transforms) -> None:
        """
        Args:
            * sub_data_dir: (str) the directory of part of the dataset whether train, val or test
            * transform: (transforms) preprocessing transforms on images
        """

        self.file_names = os.listdir(sub_data_dir)
        self.file_names = [os.path.join(sub_data_dir, file) for file in self.file_names if file.endswith(".jpg")]
        # filename ex. `2_001.jpg` as the 001 image of the class 2
        self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.file_names]
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns:
            * length: (int) total number of images loaded in this dataset instance
        """

        return len(self.file_names)

    def __getitem__(self, index: int) -> tuple(torch.Tensor, int):
        """
        Args: 
            * index: (int) index of the image from self.file_names
        
        Returns:
            * image: (Image.Image) preprocessed image
            * label: (int) ground truth label of the image
        """

        image = Image.open(self.file_names[index])
        image = self.transform(image)
        return image, self.labels[index]


def fetch_dataloader(type: str, data_dir: str, params: utils.Params) -> DataLoader:
    """
    The Torch DataLoader class is a wrapper of Dataset. It allows us to iterate through the dataset in batches, but 
    also gives us access to inbuilt functions for multiprocessing, shuffling, etc. `torch.utils.data.DataLoader` can
    take two kinds of dataset, whether a map-style dataset with `__getitem__()` and `__len__()` protocols 
    (`torch.utils.data.Dataset` for example), or a iterable-style dataset with `__iter__()` protocol 
    (`torch.utils.data.IterableDataset` for example). 

    Args:
        * type: (str) whether "train", "val" or "test"
        * data_dir: (str) the parent directory of the dataset containing train/, val/ and test/
        * params: (Params) hyperparameters

    Returns:
        * dataloader: (DataLoader) of the specified sub-dataset
    """

    assert type in ["train", "val", "test"], "Incorrect type keyword."
    sub_data_dir = os.path.join(data_dir, type)

    if type == "train":
        dataset = DatasetNameDataset(sub_data_dir, train_transformer)
        return DataLoader(dataset, 
                          batch_size=params.batch_size, 
                          shuffle=True, 
                          num_workers=params.num_workers, 
                          pin_memory=params.cuda)
    else:
        dataset = DatasetNameDataset(sub_data_dir, eval_transformer)
        return DataLoader(dataset, 
                          batch_size=params.batch_size, 
                          shuffle=False, 
                          num_workers=params.num_workers, 
                          pin_memory=params.cuda)
