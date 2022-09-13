from abc import *

import torch.utils.data as torch_data


class DatasetTemplate(torch_data.Dataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __len__():
        pass
