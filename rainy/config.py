from torch import Tensor
from torch.optim import Optimizer, RMSprop
from typing import Any, Callable, Iterable, Optional, Tuple, Union
from .net import value_net
from .net.value_net import ValueNet
from .explore import LinearCooler, Explorer, EpsGreedy
from .replay import ReplayBuffer, UniformReplayBuffer
from .util import Device, Logger
from .env_ext import ClassicalControl, EnvExt

Params = Iterable[Union[Tensor, dict]]


class Config:
    def __init__(self) -> None:
        # action/state dims are initialized lazily
        self.action_dim = 0
        self.state_dims = (0,)

        # these parameters are set really small by default,
        # so please change them manually in use
        self.batch_size = 10
        self.discount_factor = 0.99
        self.device = Device()
        self.grad_clip = 5.0
        self.max_steps = 10000
        self.replay_size = 10000
        self.train_start = 1000

        # for the cases you can't set seed in constructor, like gym.atari
        self.seed: Optional[int] = 0

        # for DQN-like algorithms
        self.double_q = False
        self.sync_freq = 200

        # logger and logging frequency
        self.logger = Logger()
        self.episode_log_freq = 100
        self.step_log_freq = 1000
        self.eval_freq = 1000
        self.save_freq = 10000
        self.save_eval_actions = False

        self.__env = lambda: ClassicalControl()
        self.__eval_env = None
        self.__exp: Callable[[ValueNet], Explorer] = \
            lambda net: EpsGreedy(1.0, LinearCooler(1.0, 0.1, 10000), net)
        self.__optim = lambda params: RMSprop(params, 0.001)
        self.__replay: Callable[[int], ReplayBuffer[Any]] = \
            lambda capacity: UniformReplayBuffer(capacity)
        self.__vn: Callable[[Tuple[int, ...], int, Device], ValueNet] = value_net.fc

    def env(self) -> EnvExt:
        env = self.__env()
        self.action_dim = env.action_dim
        self.state_dims = env.state_dims
        return env

    def set_env(self, env: Callable[[], EnvExt]) -> None:
        self.__env = env

    @property
    def eval_env(self) -> EnvExt:
        if self.__eval_env is None:
            self.__eval_env = self.env()
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env: EnvExt) -> None:
        self.action_dim = env.action_dim
        self.state_dims = env.state_dims
        self.__eval_env = env

    def explorer(self, value_net: ValueNet) -> Explorer:
        return self.__exp(value_net)

    def set_explorer(self, exp: Callable[[ValueNet], Explorer]) -> None:
        self.__exp = exp

    def optimizer(self, params: Params) -> Optimizer:
        return self.__optim(params)

    def set_optimizer(self, optim: Callable[[Params], Optimizer]) -> None:
        self.__optim = optim

    def replay_buffer(self) -> ReplayBuffer:
        return self.__replay(self.replay_size)

    def set_replay_buffer(self, replay: Callable[[int], ReplayBuffer]) -> None:
        self.__replay = replay

    def value_net(self) -> ValueNet:
        return self.__vn(self.state_dims, self.action_dim, self.device)

    def set_value_net(self, vn: Callable[[Tuple[int, ...], int, Device], ValueNet]) -> None:
        self.__vn = vn

    def policy_net(self):
        pass



