# specifies how the data should be fed to the network

import torch
from torch.autograd import Variable
from typing import Generator, List

import utils



class DataLoader(object):
    def __init__(self, data_dir: str, params: utils.Params):
        pass

    def data_iterator(self, 
                      data: dict[str, any], 
                      params: utils.Params, 
                      shuffle: bool = False) -> Generator[tuple(Variable, Variable), None, None]:
        pass

    def load_data(self, types: List[str], data_dir: str) -> dict[str, any]:
        pass