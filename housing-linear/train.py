import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
from typing import Callable, Generator, List

import utils


def train(model: nn.Module, 
          optimizer: torch.optim, 
          loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor], 
          data_iterator: Generator[tuple(torch.Tensor, torch.Tensor), None, None], 
          metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
          params: utils.Params, 
          num_steps: int):
    
    pass