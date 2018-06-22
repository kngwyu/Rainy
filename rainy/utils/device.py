import torch
from torch import Tensor
from typing import List, Optional, Union
from numpy import ndarray

class Device():
    """Utilities for handling devices
    """
    def __init__(self, gpu_limits: Optional[List[int]] = None) -> None:
        """
        :param gpu_limits: list of gpus you allow rainy to use
        """
        super().__init__()
        if gpu_limits is None:
            if torch.cuda.is_available():
                self.__use_all_gpu()
            else:
                self.__use_cpu()
        else:
            gpu_max = torch.cuda.device_count()
            self.gpu_indices = [i for i in gpu_limits if i < gpu_max]
            self.main_device = torch.device('cuda:%d' % self.gpu_indices[0])

    def __use_all_gpu(self):
        self.gpu_indices = list(range(torch.cuda.device_count()))
        self.main_device = torch.device('cuda:0')
        torch.nn.DataParallel

    def __use_cpu(self):
        self.gpu_indices = []
        self.main_device = torch.device('cpu')

    def tensor(self, x: Union[ndarray, Tensor]) -> Tensor:
        """Convert numpy array or Tensor into Tensor on main_device
        :param x: ndarray or Tensor you want to convert
        :return: Tensor
        """
        return torch.tensor(x, device=self.main_device, dtype=torch.float32)
