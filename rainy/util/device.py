from torch import nn, Tensor, torch
from typing import Iterable, List, Optional, Union
from numpy import ndarray


class Device():
    """Utilities for handling devices
    """
    def __init__(self, gpu_indices: Optional[Iterable[int]] = None) -> None:
        """
        :param gpu_limits: list of gpus you allow PyTorch to use
        """
        super().__init__()
        if gpu_indices is None:
            if torch.cuda.is_available():
                self.__use_all_gpu()
            else:
                self.__use_cpu()
        elif not gpu_indices:
            self.__use_cpu()
        else:
            self.gpu_indices = [idx for idx in gpu_indices]
            self.device = torch.device('cuda:%d' % self.gpu_indices[0])

    @property
    def unwrapped(self) -> torch.device:
        return self.device

    def tensor(self, arr: Union[ndarray, List[ndarray], Tensor], dtype=torch.float32) -> Tensor:
        """Convert numpy array or Tensor into Tensor on main_device
        :param x: ndarray or Tensor you want to convert
        :return: Tensor
        """
        t = type(arr)
        if t == Tensor:
            return arr.to(device=self.device)  # type: ignore
        elif t == ndarray or t == list:
            return torch.tensor(arr, device=self.device, dtype=dtype)
        else:
            raise ValueError('arr must be ndarray or list or tensor')

    def data_parallel(self, module: nn.Module) -> nn.DataParallel:
        return nn.DataParallel(module, device_ids=self.gpu_indices)

    def is_multi_gpu(self) -> bool:
        return len(self.gpu_indices) > 1

    def __use_all_gpu(self) -> None:
        self.gpu_indices = list(range(torch.cuda.device_count()))
        self.device = torch.device('cuda:0')

    def __use_cpu(self) -> None:
        self.gpu_indices = []
        self.device = torch.device('cpu')

    def __repr__(self) -> str:
        return str(self.device)
