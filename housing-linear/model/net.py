import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

class Net(nn.Module):
    """Linear regression module"""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=5, out_features=1)

    def forward(self, data_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            * data_batch: (torch.Tensor) features, shape: batch_size * 5

        Returns:
            * predicts: (torch.Tensor) predicted house prices, shape: batch_size * 1
        """
        return self.fc1(data_batch)


def loss_fn(predicts: torch.Tensor, targets: torch.Tensor) -> torch.FloatTensor:
    """
    Args:
        * predits: (torch.Tensor) predicted house prices, shape: batch_size * 1
        * targets: (torch.Tensor) ground truth house prices, shape: batch_size * 1

    Returns:
        * loss: (torch.FloatTensor) mean absolute error (MAE) between each element in predicts and targets
    """
    return F.l1_loss(input=predicts, target=targets)


metrics: dict[str, Callable[[np.ndarray, np.ndarray], np.float64]] = {}
