from torch import nn, Tensor, torch
from typing import List, Optional, Union
from numpy import ndarray


class Device():
    """Utilities for handling devices
    """
    def __init__(self, gpu_limits: Optional[List[int]] = None) -> None:
        """
        :param gpu_limits: list of gpus you allow PyTorch to use
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
            self.device = torch.device('cuda:%d' % self.gpu_indices[0])

    def __call__(self):
        return self.device

    def tensor(self, x: Union[ndarray, Tensor]) -> Tensor:
        """Convert numpy array or Tensor into Tensor on main_device
        :param x: ndarray or Tensor you want to convert
        :return: Tensor
        """
        return torch.tensor(x, device=self.device, dtype=torch.float32)

    def data_parallel(self, module: nn.Module) -> nn.DataParallel:
        return nn.DataParallel(module, device_ids=self.gpu_indices)

    def __use_all_gpu(self):
        self.gpu_indices = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0')

    def __use_cpu(self):
        self.gpu_indices = []
        self.device = torch.device('cpu')
