from numpy import ndarray
from torch import Tensor
from torch.optim import Optimizer, RMSprop
from typing import Any, Callable, Iterable
from .net.value_net import ValueNet, nature_dqn
from .explore import LinearCooler, Explorer, EpsGreedy
from .replay import ReplayBuffer, UniformReplayBuffer
from .util import Device, loss


class Config:
    def __init__(self) -> None:
        # initialized lazily
        self.action_dim = 0
        self.state_dim = 0

        self.batch_size = 100
        self.discount_factor = 0.01
        self.device = Device()
        self.grad_clip = 5.0
        self.replay_size = 1000
        self.seed = 0
        self.train_start = 1000

        # for value based methods
        self.__exp = lambda net: EpsGreedy(0.01, LinearCooler(0.1, 0, 100000), net)
        self.__loss_function = loss.mean_squared_loss
        self.__optim = lambda params: RMSprop(params, 0.001)
        self.__replay = lambda capacity: UniformReplayBuffer(capacity)
        self.__vn = nature_dqn
        self.__wrap_states = lambda states: states

    def explorer(self, value_net: ValueNet) -> Explorer:
        return self.__exp(value_net)

    def set_explorer(self, exp: Callable[[ValueNet], Explorer]) -> None:
        self.__exp = exp

    def loss(self, l: Tensor, r: Tensor) -> Tensor:
        return self.__loss_function(l, r)

    def set_loss(self, loss: Callable[[Tensor, Tensor], Tensor]) -> None:
        self.__loss_function = loss

    def optimizer(self, params: Iterable[Any]) -> Optimizer:
        return self.__optim(params)

    def set_optimizer(self, optim: Callable[[Iterable[Any]], Optimizer]) -> None:
        self.__optim = optim

    def replay_buffer(self) -> ReplayBuffer:
        return self.__replay(self.replay_size)

    def set_replay_buffer(self, replay: Callable[[int], ReplayBuffer]) -> None:
        self.__replay = replay

    def value_net(self) -> ValueNet:
        return self.__vn(self.state_dim, self.action_dim, self.device)

    def set_value_net(self, vn: Callable[[int, int, Device], ValueNet]) -> None:
        self.__vn = vn

    def wrap_states(self, state: ndarray) -> ndarray:
        return self.__wrap_states(state)

    def set_wrap_states(self, wrap_states: Callable[[ndarray], ndarray]) -> None:
        self.__wrap_states = wrap_states

    def set(self, key: str, value: Any):
        self.__setattr__(key, value)

    def policy_net(self):
        pass



