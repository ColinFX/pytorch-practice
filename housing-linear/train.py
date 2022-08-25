import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
from typing import Callable, Generator, List

import utils


def train(model: nn.Module, 
          optimizer: torch.optim.optimizer.Optimizer, 
          loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor], 
          data_iterator: Generator[tuple(torch.Tensor, torch.Tensor), None, None], 
          metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
          params: utils.Params, 
          num_steps: int):
    """
    Train the model on `num_steps` batches/iterations of size `params.batch_size` as one epoch. 
    
    Args:
        * model: (nn.Module) the neural network
        * optimizer: (torch.optim) the optimizer for parameters in the model
        * loss_fn: (Callable) output_batch, labels_batch -> loss
        * data_ietrator: (Generator) -> train_batch, labels_batch
        * metrics: (dict) metric_name -> (function (Callable) output_batch, labels_batch -> metric_value)
        * params: (utils.Params) hyperparameters
        * num_steps: (int) number of batches to train for each epoch
    """

    model.train()
    summ: List[dict[str, float]] = []
    loss_avg = utils.RunningAverage()

    t = trange(num_steps)
    for i in t:
        train_batch, targets_batch = next(data_iterator)
        predicts_batch = model(train_batch)
        loss = loss_fn(predicts_batch, targets_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
