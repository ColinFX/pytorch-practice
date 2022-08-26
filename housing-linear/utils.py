import json
import logging
import os


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