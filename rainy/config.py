from numpy import ndarray
from torch import Tensor
from torch.optim import Optimizer, RMSprop
from typing import Any, Callable, Iterable, Optional
from .net import value_net
from .net.value_net import ValueNet
from .explore import LinearCooler, Explorer, EpsGreedy
from .replay import ReplayBuffer, UniformReplayBuffer
from .util import Device
from .env_ext import ClassicalControl, EnvExt


class Config:
    def __init__(self) -> None:
        self.action_dim = 0
        self.state_dim = 0
        self.batch_size = 100
        self.discount_factor = 0.01
        self.device = Device()
        self.double_q = False
        self.grad_clip = 5.0
        self.max_steps = 100000
        self.replay_size = 1000
        self.seed = 0
        self.sync_freq = 1000
        self.train_start = 1000

        self.__env = lambda: ClassicalControl()
        self.__exp = lambda net: EpsGreedy(0.01, LinearCooler(0.1, 0, 100000), net)
        self.__optim = lambda params: RMSprop(params, 0.001)
        self.__replay = lambda capacity: UniformReplayBuffer(capacity)
        self.__vn = value_net.fc
        self.__wrap_states = lambda states: states

    def env(self) -> EnvExt:
        env = self.__env()
        self.action_dim = env.action_dim
        self.state_dim = env.state_dim
        return env

    def set_env(self, env: Callable[[], EnvExt]) -> None:
        self.__env = env

    def explorer(self, value_net: ValueNet) -> Explorer:
        return self.__exp(value_net)

    def set_explorer(self, exp: Callable[[ValueNet], Explorer]) -> None:
        self.__exp = exp

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



