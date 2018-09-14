from gym import Env
from numpy import ndarray
from torch.optim import Optimizer, RMSprop
from typing import Any, Callable, Iterable
from .lib.device import Device
from .net.value_net import ValueNet
from .explore import LinearCooler, Explorer, Greedy
from .replay import ReplayBuffer, UniformReplayBuffer


class Config:
    def __init__(self) -> None:
        # for value based methods
        self.gpu_limits = [0]
        self.state_dim = 1000
        self.action_dim = 1000
        self.replay_size = 1000
        self.exp_steps = 1000
        self.seed = 0
        self.__exp_gen = lambda net: Greedy(0.01, LinearCooler(0.1, 0, 100000) ,net)
        self.__optim_gen = lambda params: RMSprop(params, 0.001)
        self.__replay_gen = lambda capacity: UniformReplayBuffer(capacity)
        self.__task_gen = None
        self.__vn_gen = None
        self.__wrap_state = lambda state: state

    def device(self) -> Device:
        return Device(gpu_limits=self.gpu_limits)

    def env(self) -> Env:
        self.__env_gen()

    def set_env(self, env: Callable[[], Env]) -> None:
        self.__env_gen = env

    def explorer(self, value_net: ValueNet) -> Explorer:
        return self.__exp_gen(value_net)

    def set_explorer(self, exp_gen: Callable[[ValueNet], Explorer]) -> None:
        self.__exp_gen = exp_gen

    def optimizer(self, params: Iterable[Any]) -> Optimizer:
        return self.__optim_gen(params)

    def set_optimizer(self, optim_gen: Callable[[Iterable[Any]], Optimizer]) -> None:
        self.__optim_gen = optim_gen

    def replay_buffer(self) -> ReplayBuffer:
        return self.__replay_gen(self.replay_size)

    def set_replay_buffer(self, replay_gen: Callable[[int], ReplayBuffer]) -> None:
        self.__replay_gen = replay_gen

    def value_net(self, state_dim: int, action_dim: int) -> ValueNet:
        device = self.device()
        return self.__vn_gen(state_dim, action_dim, device)

    def set_value_net(self, vn_gen: Callable[[int, int, Device], ValueNet]) -> None:
        self.__vn_gen = vn_gen

    def wrap_state(self, state: ndarray) -> ndarray:
        return self.__wrap_state(state)
    
    def set_wrap_state(self, wrap_state: Callable[[ndarray], ndarray]) -> None:
        self.__wrap_state = wrap_state

    def policy_net(self):
        pass



