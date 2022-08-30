from typing_extensions import Self
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """CNN module"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=7*7*32, out_features=10)

    def forward(self, data_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            * data_batch: (torch.Tensor) contains a batch of images, shape: batch_size * 1 * 28 * 28
        
        Returns:
            * out: (torch.Tensor) predicted log probability distribution of each image, shape: batch_size * 10
        """

        data_batch = self.conv1(data_batch)     # batch_size * 16 * 28 * 28
        data_batch = F.relu(data_batch)
        data_batch = self.mp1(data_batch)       # batch_size * 16 * 14 * 14
        data_batch = self.conv2(data_batch)     # batch_size * 32 * 14 * 14
        data_batch = F.relu(data_batch)
        data_batch = self.mp2(data_batch)       # batch_size * 16 * 7 * 7

        # flatten
        data_batch = data_batch.view(-1, 7*7*32)    # batch_size * 7*7*16
        
        data_batch = self.fc1(data_batch)       # batch_size * 10
        return data_batch

    def reset_weights(self):
        """Reset all weights before next fold to avoid weight leakage"""

        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()


def loss_fn(outputs: torch.Tensor, labels: torch.Tensor) -> torch.FloatTensor:
    """
    Args:
        * outputs: (torch.Tensor) outpout of the model, shape: batch_size * 10
        * labels: (torch.Tensor) ground truth label of the image, shape: batch_size with each element a value in [0:9]

    Returns:
        * loss: (torch.FloatTensor) cross entropy loss for all images in the batch
    """
    loss = nn.CrossEntropyLoss()
    return loss(outputs, labels)


def accuracy(outputs: np.ndarray, labels: np.ndarray) -> np.float64:
    """
    Args: 
        * outputs: (np.ndarray) outpout of the model, shape: batch_size * 6
        * labels: (np.ndarray) ground truth label of the image, shape: batch_size with each element a value in [0:9]

    Returns:
        * accuracy: (float) in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels) / float(labels.size)


metrics = {"accuracy": accuracy}
