"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
from typing import Callable, Generator, List

from model.data_loader import fetch_dataloader
import model.net as net
from evaluate import evaluate
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data", help="")
parser.add_argument("--model_dir", default="experiments/base_model", help="")   # hyper-parameter json file
parser.add_argument("--restore_file", default=None, help="")    # "best" or "last", model weights checkpoint


def train(model: nn.Module, 
          optimizer: torch.optim.Optimizer, 
          loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor], 
          data_iterator: Generator[tuple[torch.Tensor, torch.Tensor], None, None], 
          metrics: dict[str, Callable[[np.ndarray, np.ndarray], np.float64]], 
          params: utils.Params, 
          num_steps: int):
    """
    Train the model on `num_steps` batches/iterations of size `params.batch_size` as one epoch. 
    
    Args:
        * model: (nn.Module) the neural network
        * optimizer: (torch.optim.Optimizer) the optimizer for parameters in the model
        * loss_fn: (Callable) output_batch, labels_batch -> loss
        * data_ietrator: (Generator) -> train_batch, labels_batch
        * metrics: (dict) metric_name -> (function (Callable) output_batch, labels_batch -> metric_value)
        * params: (utils.Params) hyperparameters
        * num_steps: (int) number of batches to train for each epoch
    """

    model.train()   # set model to training mode
    summ: List[dict[str, float]] = []   # summary of metrics for the epoch
    loss_avg = utils.RunningAverage()   # running average of loss for the epoch
    
    t = trange(num_steps)
    for i in t:
        train_batch, labels_batch = next(data_iterator)

        if params.cuda:     # move to GPU if available
            train_batch = train_batch.to(device=torch.device("cuda"), non_blocking=True)
            labels_batch = labels_batch.to(device=torch.device("cuda"), non_blocking=True)

        # core pipeline
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)
        optimizer.zero_grad()   # clear previous gradients
        loss.backward()     # compute gradients of all variables
        optimizer.step()    # update weights using calculated gradients

        # evaluate summaries once in a while
        if i % params.save_summary_steps == 0:
            output_batch = output_batch.detach().to(device=torch.device("cpu")).numpy()
            labels_batch = labels_batch.detach().to(device=torch.device("cpu")).numpy()
            summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
            summary_batch["loss"] = loss.item()
            summ.append(summary_batch)

        loss_avg.update(loss.item())
        t.set_postfix(loss="{:05.3f}".format(loss_avg()))

    metrics_mean = {metric: np.mean([batch[metric] for batch in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(key,value) for key,value in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model: nn.Module, 
                       optimizer: torch.optim.Optimizer, 
                       loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor], 
                       train_data_loader: DataLoader, 
                       val_data_loader: DataLoader, 
                       metrics: dict[str, Callable[[np.ndarray, np.ndarray], np.float64]], 
                       params: utils.Params, 
                       model_dir: str, 
                       restore_file: str = None) -> dict[str, np.float64]: 
    """
    Train and evaluate the model on `params.num_epochs` epochs and save checkpoints and metrics. 
    
    Args:
        * model: (nn.Module) the neural network
        * optimizer: (torch.optim.Optimizer) the optimizer for parameters in the model
        * loss_fn: (Callable) output_batch, labels_batch -> loss
        * train_data_loader: (DalaLoader) for training set
        * val_data_loader: (DalaLoader) for validation set
        * metrics: (dict) metric_name -> (function (Callable) output_batch, labels_batch -> metric_value)
        * params: (utils.Params) hyperparameters
        * model_dir: (str) directory containing config, checkpoints and log
        * restore_file: (str) optional - name of checkpoint to restore from (without extension .pth.tar)

    Returns:
        * metrics: (dict) metric_name -> metric_value on the val set of the best epoch
    """

    # reload weights from checkpoint is available
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file+".pth.tar")
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = 0.0

    best_metrics: dict[str, np.float64] = []

    for epoch in range(params.num_epochs):
        logging.info("Epoch {}/{}".format(epoch+1, params.num_epochs))
        
        # train
        num_steps = int((params.train_size+1) // params.batch_size)
        train_data_iterator = iter(train_data_loader)
        train(model, optimizer, loss_fn, train_data_iterator, metrics, params, num_steps)

        # evaluate
        num_steps = int((params.val_size+1) // params.batch_size)
        val_data_iterator = iter(val_data_loader)
        val_metrics = evaluate(model, loss_fn, val_data_iterator, metrics, params, num_steps)
        val_acc = val_metrics["accuracy"]
        is_best = (val_acc>=best_val_acc)

        # save weights checkpoint
        utils.save_checkpoint({"epoch": epoch+1, 
                               "state_dict": model.state_dict(), 
                               "optim_dict": optimizer.state_dict()}, 
                              is_best=is_best, 
                              checkpoint_dir=model_dir)
        
        # overwrite best metrics evaluation result if the model is the best by far
        if is_best:
            best_metrics = val_metrics
            logging.info("- Found new best accuracy")
            best_val_acc = val_acc
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # overwrite last metrics evaluation result
        last_json_path = os.path.join(model_dir, "metric_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    return best_metrics


if __name__ == "__main__":
    """Train the model on the train and val set"""

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(json_path), "No json file found at {}.".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    
    # set random seed
    torch.manual_seed(42)
    if params.cuda:
        torch.cuda.manual_seed(42)
    
    # set logger
    utils.set_logger(os.path.join(args.model_dir, "train.log"))
    logging.info("Loading the dataset...")

    # load data
    train_data_loader = fetch_dataloader("train", args.data_dir, params)
    val_data_loader = fetch_dataloader("test", args.data_dir, params)
    params.train_size = len(train_data_loader.dataset)
    params.val_size = len(val_data_loader.dataset)
    logging.info("- Done")

    # train and evaluate pipeline
    model = net.Net().to(device=torch.device("cuda")) if params.cuda else net.Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = net.loss_fn
    metrics = net.metrics
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, optimizer, loss_fn, train_data_loader, val_data_loader, metrics, params, args.model_dir, 
                       args.restore_file)
