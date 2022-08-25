import os
from typing import List

import pandas as pd
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split


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
    dataframe.drop(["address"], axis=1, inplace=True)
    
    # split input and target and convert to tensor
    input_columns = dataframe.columns[:-1]
    target_column = dataframe.columns[-1]
    input_array = dataframe.copy(deep=True)[input_columns].to_numpy()
    target_array = dataframe.copy(deep=True)[target_column].to_numpy().reshape((5000, 1))
    input_tensor = torch.from_numpy(input_array)
    target_tensor = torch.from_numpy(target_array)

    return TensorDataset(input_tensor, target_tensor)


def fetch_dataloaders(data_dir: str, val_split_percent: float, batch_size: int) -> dict[str, DataLoader]:
    """
    Args:
        * data_dir: (str) directory containing USA_Housing.csv file
        * params

    Returns:
        * dataloaders: (dict) "train" or "val" -> DataLoader
    """

    # split dataset
    dataset = fetch_dataset(data_dir)
    val_size = len(dataset) * val_split_percent
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)
    return {"train": train_dataloader, "val": val_dataloader}
