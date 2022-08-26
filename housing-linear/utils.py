import json
import logging
import os

import torch
import torch.nn as nn


class Params(object):
    """
    Example:
        * create params instance by `params = Params(json_path)`
        * change one param value by `params.learning_rate = 0.5`
        * show one param values by `print(params.learning_rate)`
    """

    def __init__(self, json_path: str):
        with open(json_path) as file:
            params = json.load(file)
            self.__dict__.update(params)

    def save(self, json_path: str):
        with open(json_path, 'w') as file:
            json.dump(self.__dict__, file, indent=4)

    def update(self, json_path: str):
        with open(json_path) as file:
            params = json.load(file)
            self.__dict__.update(params)

    @property
    def dict(self):
        """dict-like access to param by `params.dict["learning_rate"]`"""
        return self.__dict__


class RunningAverage():
    """
    Examples:
        * create running_avg instance by `running_avg = RunningAverage()`
        * add new item by `running_avg.update(2.0)`
        * get current average by `running_avg()`
    """

    def __init__(self):
        self.steps: int = 0
        self.total: float = 0

    def update(self, val: float):
        self.steps += 1
        self.total += val

    def __call__(self) -> float:
        return self.total / float(self.steps)


def set_logger(log_path: str):
    """
    Args:
        log_path: (str) to store log files

    Examples:
        * write log by `logging.info("Start training...")`
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # logging to the file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(dictionary: dict[str, float], json_path: str):
    """
    Save hyperparameter dict to json file. 

    Args:
        dictionary: (dict) of hyperparameters
        json_path: (str) path to save json file
    """

    with open(json_path, 'w') as file:
        # convert the values to float for json since it doesn't accept np.array, np.float, etc
        dictionary = {key: float(value) for key, value in dictionary.items()}
        json.dump(dictionary, file, indent=4)


def save_checkpoint(state: dict[str, float or dict], is_best: bool, checkpoint_dir: str):
    """
    Args: 
        * state: (dict) containing "model" key and maybe "epoch" and "optimizer", the value for "model" and "optimizer" 
          is a python dictionary object that maps each layer to its parameter tensor
        * is_best: (bool) whether the model is the best till that moment
        * checkpoint_dir: (str) folder to save weights
    """

    # create folder if not exist
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist, making directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    else:
        print("Checkpoint Directory exists")
    
    torch.save(state, os.path.join(checkpoint_dir, "last.pth.tar"))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, "best.pth.tar"))


def load_checkpoint(checkpoint_path: str, 
                    model: nn.Module, 
                    optimizer: torch.optim.optimizer.Optimizer = None) -> dict[str, float or dict]:
    """
    Args: 
        * checkpoint_dir: (str) path of the checkpoint
        * model: (nn.Module) model that weights will be loaded to
        * optimizer: (torch.optim.optimizer.Optimizer) optional - optimizer that weights will be loaded to
    """

    if not os.path.exists(checkpoint_path):
        raise("File doesn't exist {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])
    return checkpoint
