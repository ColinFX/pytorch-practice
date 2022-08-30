"""Cross validate the model"""

import argparse
import logging
import os

import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from typing import Callable

from train import train_and_evaluate
from model.data_loader import fetch_dataset
import model.net as net
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data", help="")
parser.add_argument("--model_dir", default="experiments/base_model", help="")   # hyper-parameter json file
parser.add_argument("--restore_file", default=None, help="")    # "best" or "last", model weights checkpoint


def cross_validate(model: net.Net, 
                   optimizer: torch.optim.Optimizer, 
                   loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.FloatTensor], 
                   dataset: Dataset, 
                   metrics: dict[str, Callable[[np.ndarray, np.ndarray], np.float64]], 
                   params: utils.Params, 
                   model_dir: str, 
                   restore_file: str = None): 
    """
    Run k-fold cross validation on whole train set as sub train set + val set to evaluate the model under current 
    network structure and hyperparameters. Average metrics result will be saved to json file.
    
    Args:
        * model: (nn.Module) the neural network
        * optimizer: (torch.optim.Optimizer) the optimizer for parameters in the model
        * loss_fn: (Callable) output_batch, labels_batch -> loss
        * dataset: (Dataset) the overall dataset to be splited into sub train set and val set in k-fold way
        * metrics: (dict) metric_name -> (function (Callable) output_batch, labels_batch -> metric_value)
        * params: (utils.Params) hyperparameters
        * model_dir: (str) directory containing config, checkpoints and log
        * restore_file: (str) optional - name of checkpoint to restore from (without extension .pth.tar)
    """

    k_fold = KFold(n_splits=6, shuffle=True)    # set k-fold cross validator
    avg_metrics = {metric_name: utils.RunningAverage() for metric_name in metrics.keys()}
    avg_metrics["loss"] = utils.RunningAverage()

    for fold, (train_ids, val_ids) in enumerate(k_fold.split(dataset)):
        logging.info("Fold {}/6".format(fold+1))

        # split train and val sub set
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        train_data_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=train_subsampler)
        val_data_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=val_subsampler)

        # train and evaluate the model under current train-val split
        model.reset_weights()
        logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
        fold_metrics = train_and_evaluate(model, optimizer, loss_fn, train_data_loader, val_data_loader, metrics, params, 
                                     args.model_dir, args.restore_file)

        # update avg_metrics
        for metric_name in avg_metrics.keys():
            avg_metrics[metric_name].update(fold_metrics[metric_name])

    # save to json
    avg_metrics_path = os.path.join(model_dir, "average_metrics_cross_validation.json")
    avg_metrics = {metric_name: avg_metrics[metric_name]() for metric_name in avg_metrics.keys()}
    utils.save_dict_to_json(avg_metrics, avg_metrics_path)

        
if __name__ == "__main__":
    """Run cross validation on the train set"""

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
    utils.set_logger(os.path.join(args.model_dir, "cross_validate.log"))
    logging.info("Loading the dataset...")

    # load data
    dataset = fetch_dataset("train", args.data_dir)
    params.train_size = len(dataset) * 5/6
    params.val_size = len(dataset) * 1/6

    # cross validation pipeline
    model = net.Net().to(device=torch.device("cuda")) if params.cuda else net.Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate)
    loss_fn = net.loss_fn
    metrics = net.metrics
    logging.info("Starting cross validation for 6 folds.".format(params.num_epochs))
    cross_validate(model, optimizer, loss_fn, dataset, metrics, params, args.model_dir, args.restore_file)
