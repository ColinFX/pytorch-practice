import os

import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split

import utils


def fetch_dataset(data_dir: str) -> Dataset:
    """
    USA_Housing dataset is a csv file containing 5000 rows with header ['Avg. Area Income', 'Avg. Area House Age', 
    'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population', 'Price', 'Address']. 

    Args:
        * data_dir: (str) directory containing USA_Housing.csv file
    
    Returns:
        * dataset: (Dataset) of all the data, not splited into train and val yet
    """

    # pandas preprocessing
    dataframe = pd.read_csv(os.path.join(data_dir, "USA_Housing.csv"))
    dataframe.drop(["Address"], axis=1, inplace=True)
    print(dataframe.info())
    
    # split input and target and convert to tensor
    input_columns = dataframe.columns[:-1]
    target_column = dataframe.columns[-1]
    input_array = dataframe.copy(deep=True)[input_columns].to_numpy()
    target_array = dataframe.copy(deep=True)[target_column].to_numpy().reshape((5000, 1))
    input_tensor = torch.from_numpy(input_array)
    target_tensor = torch.from_numpy(target_array)

    return TensorDataset(input_tensor, target_tensor)


def fetch_dataloaders(data_dir: str, params: utils.Params) -> dict[str, DataLoader]:
    """
    Args:
        * data_dir: (str) directory containing USA_Housing.csv file
        * params

    Returns:
        * dataloaders: (dict) "train", "val", "test" -> DataLoader
    """

    # split dataset
    dataset = fetch_dataset(data_dir)
    val_size = int(len(dataset) * params.val_split_percentage)
    test_size = int(len(dataset) * params.test_split_percentage)
    train_size = int(len(dataset) - val_size - test_size)
    train_dataset, val_dataset, test_dataset = random_split(dataset, lengths=[train_size, val_size, test_size])

    train_dataloader = DataLoader(train_dataset, params.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, params.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, params.batch_size, shuffle=False)
    return {"train": train_dataloader, "val": val_dataloader, "test": test_dataloader}
