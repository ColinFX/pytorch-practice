# utility functions for handling hyperparams/logging/storing model

import json

import torch



class Params(object):
    # create params instance by `params = Params(json_path)`
    # change one param value by `params.learning_rate = 0.5`
    # show all params values by `print(params.learning_rate)`

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
        # dict-like access to param by `params.dict["learning_rate"]`
        return self.__dict__


class RunningAverage():
    # create running_avg instance by `running_avg = RunningAverage()`
    # add new item by `running_avg.update(2.0)`
    # get current average by `running_avg()`

    def __init__(self):
        self.steps: int = 0
        self.total: float = 0

    def udpate(self, val: float):
        self.steps += 1
        self.total += val

    def __call__(self) -> float:
        return self.total / float(self.steps)


def set_logger(logger_path: str):
    pass


def save_dict_to_json(dictionary: dict[str, float], json_path: str):
    pass


def save_checkpoint(state: dict, is_best: bool, checkpoint_path: str):
    pass


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim = None):
    pass
