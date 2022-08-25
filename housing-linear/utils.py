import json
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