"""Define neural network, loss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils


class Net(nn.Module):
    def __init__(self, params: utils.Params):
        """
        This example is a classification task of RGB images of resolution 64*64 on 6 different classes.

        The neural network is composed of:
            * an embedding layer to convert input data into vectors
            * convolutional layers along with batch normalization, maxpooling and relu
            * fully connected layers to convert the output of each image to a distribution over 6 classes
        
        Args:
            * params: (utils.Params) to pass num_channels and dropout_rate
        """
        super(Net, self).__init__()
        self.num_channels: int = params.num_channels
        self.dropout_rate: float = params.dropout_rate
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=self.num_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.num_channels)
        self.conv2 = nn.Conv2d(in_channels=self.num_channels, out_channels=self.num_channels*2, kernel_size=3, stride=1, 
                               padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=self.num_channels*2)
        self.conv3 = nn.Conv2d(in_channels=self.num_channels*2, out_channels=self.num_channels*4, kernel_size=3, 
                               stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=self.num_channels*4)

        self.fc1 = nn.Linear(in_features=8*8*self.num_channels*4, out_features=self.num_channels*4)
        self.fcbn1 = nn.BatchNorm1d(num_features=self.num_channels*4)
        self.fc2 = nn.Linear(in_features=self.num_channels*4, out_features=6)

    def forward(self, data_batch: Variable) -> Variable:
        """
        Args:
            * data_batch: (Variable) contains a batch of images, shape: batch_size * 3 * 64 * 64
        
        Returns:
            * out: (Variable) predicted log probability distribution of each image, shape: batch_size * 6
        """
        data_batch = self.bn1(self.conv1(data_batch))       # batch_size * num_channels * 64 * 64
        data_batch = F.relu(F.max_pool2d(data_batch, 2))    # batch_size * num_channels * 32 * 32
        data_batch = self.bn2(self.conv2(data_batch))       # batch_size * num_channels*2 * 32 * 32
        data_batch = F.relu(F.max_pool2d(data_batch, 2))    # batch_size * num_channels*2 * 16 * 16
        data_batch = self.bn3(self.conv3(data_batch))       # batch_size * num_channels*4 * 16 * 16
        data_batch = F.relu(F.max_pool2d(data_batch, 2))    # batch_size * num_channels*4 * 8 * 8

        # flatten
        data_batch = data_batch.view(-1, 8*8*self.num_channels*4)   # batch_size * 8*8*num_chennels*4

        data_batch = F.dropout(F.relu(self.fcbn1(self.fc1(data_batch))), p=self.dropout_rate, training=self.training)
                                                # batch_size * num_channels*4
        data_batch = self.fc2(data_batch)       # batch_size * 6
        return F.log_softmax(data_batch, dim=1) # batch_size * 6


def loss_fn(outputs: Variable, labels: Variable) -> Variable:
    """
    Args:
        * outputs: (Variable) outpout of the model, shape: batch_size * 6
        * labels: (Variable) ground truth label of the image, shape: batch_size with each element a value in 
            [0,1,2,3,4,5]

    Returns:
        * loss: (Variable) cross entropy loss for all images in the batch
    """
    loss = nn.CrossEntropyLoss()
    return loss(outputs, labels)


def accuracy(outputs: np.ndarray, labels: np.ndarray) -> float:
    """
    Args: 
        * outputs: (np.ndarray) outpout of the model, shape: batch_size * 6
        * labels: (np.ndarray) ground truth label of the image, shape: batch_size with each element a value in 
            [0,1,2,3,4,5]

    Returns:
        * accuracy: (float) in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels) / float(labels.size)


metrics = {"accuracy": accuracy}
