from typing import List, Union

import torch
from numpy import ndarray
from torch import LongTensor, Tensor, nn

from ..lib import mpi
from ..prelude import ArrayLike, Self


class Device:
    """Utilities for handling devices
    """

    def __init__(self, use_cpu: bool = False, gpu_indices: List[int] = []) -> None:
        """
        :param gpu_limits: list of gpus you allow PyTorch to use
        """
        super().__init__()
        if use_cpu or not torch.cuda.is_available():
            self.__use_cpu()
        else:
            self.gpu_indices = gpu_indices if gpu_indices else self.__all_gpu()
            self.device = torch.device("cuda:{}".format(self.gpu_indices[0]))

    def split(self) -> Self:
        """If self has multiple GPUs, split them into two devices.
        """
        num_gpus = len(self.gpu_indices)
        if num_gpus < 2:
            return self
        else:
            index = num_gpus // 2
            gpu_indices = self.gpu_indices[index:]
            self.gpu_indices = self.gpu_indices[:index]
            return Device(gpu_indices=gpu_indices)

    @property
    def unwrapped(self) -> torch.device:
        return self.device

    def tensor(self, arr: ArrayLike, dtype: torch.dtype = torch.float32) -> Tensor:
        """Convert numpy array or PyTorch Tensor into Tensor on main_device
        :param x: ndarray or Tensor you want to convert
        :return: Tensor
        """
        t = type(arr)
        if t is Tensor:
            return arr.to(device=self.device)  # type: ignore
        elif t is ndarray or t is list:
            return torch.tensor(arr, device=self.device, dtype=dtype)
        else:
            raise ValueError("arr must be ndarray or list or tensor")

    def zeros(
        self, size: Union[int, tuple], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        return torch.zeros(size, device=self.device, dtype=dtype)

    def ones(
        self, size: Union[int, tuple], dtype: torch.dtype = torch.float32
    ) -> Tensor:
        return torch.ones(size, device=self.device, dtype=dtype)

    def data_parallel(self, module: nn.Module) -> nn.DataParallel:
        return nn.DataParallel(module, device_ids=self.gpu_indices)

    def is_multi_gpu(self) -> bool:
        return len(self.gpu_indices) > 1

    def indices(self, end: int, start: int = 0) -> LongTensor:
        res = torch.arange(start=start, end=end, device=self.device, dtype=torch.long)
        return res  # type: ignore

    def __all_gpu(self) -> List[int]:
        ngpus = torch.cuda.device_count()
        size, rank = mpi.local_size_and_rank()
        if size > 1:
            if ngpus < size:
                raise ValueError(f"Too many local processes {size} for {ngpus} gpus")
            return [rank]
        return list(range(ngpus))

    def __use_cpu(self) -> None:
        self.gpu_indices = []
        self.device = torch.device("cpu")

    def __repr__(self) -> str:
        return str(self.device)
