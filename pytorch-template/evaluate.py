"""Evaluate the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import trange
from typing import Callable, Generator, List

from model.data_loader import fetch_dataloader
import model.net as net
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data/dataset_name", help="")
parser.add_argument("--model_dir", default="experiments/base_model", help="")   # hyper-parameter json file
parser.add_argument("--restore_file", default="best", help="")    # "best" or "train", model weights checkpoint ?


def evaluate(model: nn.Module, 
             loss_fn: Callable[[Variable, Variable], Variable], 
             data_iterator: Generator[tuple(Variable, Variable), None, None], 
             metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]], 
             params: utils.Params, 
             num_steps: int) -> dict[str, float]:
    """
    Evaluate the model on `num_steps` batches/iterations of size `params.batch_size` as one epoch.
    
    Args:
        * model: (nn.Module) the neural network
        * loss_fn: (Callable) output_batch, labels_batch -> loss
        * data_ietrator: (Generator) -> train_batch, labels_batch
        * metrics: (dict) of functions (Callable) output_batch, labels_batch -> metric
        * params: (utils.Params) hyperparameters
        * num_steps: (int) number of batches to train for each epoch
    """
    
    model.eval()   # set model to evaluation mode
    summ: List[dict[str, float]] = []   # summary for the epoch
    
    t = trange(num_steps)
    for i in t:
        data_batch, labels_batch = next(data_iterator)

        # core pipeline
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch)

        # evaluate all metrics on the batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch) for metric in metrics}
        summary_batch["loss"] = loss.item()
        summ.append(summary_batch)

    metrics_mean = {metric: np.mean([batch[metric] for batch in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(key,value) for key,value in metrics_mean.items())
    logging.info("- Eval metrics: " + metrics_string)
    return metrics_mean


if __name__ == "__main__":
    """Evaluate the model on the test set"""

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
    utils.set_logger(os.path.join(args.model_dir, "evaluate.log"))
    logging.info("Loading the dataset...")

    # load data
    data_loader = DatasetNameDataLoader(args.data_dir, params)
    data = data_loader.load_data(["test"], args.data_dir)
    test_data = data["test"]
    params.test_size = test_data["size"]
    test_data_iterator = data_loader.data_iterator(test_data, params)
    logging.info("- Done")

    # define model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    loss_fn = net.loss_fn
    metrics = net.metrics
    num_steps = (params.test_size+1) // params.batch_size

    # evaluate pipeline
    logging.info("Starting evaluation")
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file+".pth.tar"), model)    
    test_metrics = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps)

    # save metrics evaluation result on the restore_file
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
