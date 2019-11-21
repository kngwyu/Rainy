import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Tuple
from ..prelude import Array, Params

try:
    import horovod.torch as hvd

    hvd.init()
    IS_MPI_ROOT = hvd.rank() == 0

    if hvd.size() == 1:
        raise ModuleNotFoundError("When hvd.size() == 1 we do not use horovod")

    def setup_models(*args) -> None:
        for model in args:
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    def setup_optimizer(opt: Optimizer) -> Optimizer:
        hvd.broadcast_optimizer_state(opt, root_rank=0)
        return hvd.DistributedOptimizer(opt)

    def clip_and_step(params: Params, max_norm: float, opt: Optimizer) -> None:
        opt.synchronize()
        torch.nn.utils.clip_grad_norm_(params, max_norm)
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

    def array_mean_and_var(arr: Array) -> Tuple[Array, Array]:
        m, v = tensor_mean_and_var(torch.from_numpy(arr))
        return m.numpy(), v.numpy()

    def tensor_mean_and_var(t: Tensor) -> Tuple[Tensor, Tensor]:
        mean = hvd.allreduce_(t.mean(dim=0))
        var = hvd.allreduce_((t - mean).pow(2).mean(dim=0))
        return mean, var


except ModuleNotFoundError:

    IS_MPI_ROOT = True

    def setup_models(*args) -> None:
        pass

    def setup_optimizer(opt: Optimizer) -> Optimizer:
        return opt

    def broadcast_model(model: torch.nn.Module) -> None:
        pass

    def local_size_and_rank() -> Tuple[int, int]:
        return 1, 0

    def clip_and_step(params: Params, max_norm: float, opt: Optimizer) -> None:
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        opt.step()

    def global_size() -> int:
        return 1

    def array_reduce_(arr: Array, average: bool = True) -> None:
        pass

    def array_mean_and_var(arr: Array) -> Tuple[Array, Array]:
        return arr.mean(axis=0), arr.var(axis=0)

    def tensor_mean_and_var(t: Tensor) -> Tuple[Tensor, Tensor]:
        return t.mean(dim=0), t.var(dim=0, unbiased=False)
