import torch
from torch.optim import Optimizer
from typing import Tuple
from ..prelude import Array
try:
    import horovod.torch as hvd
    hvd.init()
    IS_MPI_ROOT = hvd.rank() == 0

    def setup(model: torch.nn.Module, opt: Optimizer) -> Optimizer:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(opt, root_rank=0)
        return hvd.DistributedOptimizer(opt, model.named_parameters())

    def clip_and_step(model: torch.nn.Module, max_norm: float, opt: Optimizer) -> None:
        opt.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        with opt.skip_synchronize():
            opt.step()

    def broadcast_model(model: torch.nn.Module) -> None:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    def local_size_and_rank() -> Tuple[int, int]:
        return hvd.local_size(), hvd.local_rank()

    def global_size() -> int:
        return hvd.size()

    def array_reduce_(arr: Array, average: bool = True) -> None:
        t = torch.from_numpy(arr)
        hvd.allreduce_(t, average=average)

except ModuleNotFoundError:

    IS_MPI_ROOT = True

    def setup(model: torch.nn.Module, opt: Optimizer) -> Optimizer:
        return opt

    def broadcast_model(model: torch.nn.Module) -> None:
        pass

    def local_size_and_rank() -> Tuple[int, int]:
        return 1, 0

    def clip_and_step(model: torch.nn.Module, max_norm: float, opt: Optimizer) -> None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        opt.step()

    def global_size() -> int:
        return 1

    def array_reduce_(arr: Array, average: bool = True) -> None:
        pass
