from torch import nn, Tensor
from torch.optim import Optimizer, RMSprop
from typing import Callable, Dict, Iterable, Optional, Tuple, Union
from .net import actor_critic, ValuePredictor, value
from .explore import DummyCooler, Cooler, LinearCooler, Explorer, EpsGreedy
from .replay import DqnReplayFeed, ReplayBuffer, UniformReplayBuffer
from .util import Device, Logger
from .envs import ClassicalControl, DummyParallelEnv, EnvExt, ParallelEnv

Params = Iterable[Union[Tensor, dict]]
NetFn = Callable[[Tuple[int, ...], int, Device], nn.Module]


class Config:
    def __init__(self) -> None:
        # action/state dims are initialized lazily
        self.action_dim = 0
        self.state_dim: Tuple[int, ...] = (0,)

        # common parameters
        self.discount_factor = 0.99
        self.device = Device()
        self.grad_clip = 5.0  # I recommend 0.5 for A2C
        self.max_steps = 10000
        self.eval_deterministic = True

        # For replay buffer algorithms
        self.replay_batch_size = 10
        self.replay_size = 10000
        self.train_start = 1000

        # For the cases you can't set seed in constructor, like gym.atari
        self.seed: Optional[int] = 0

        # For DQN-like algorithms
        self.double_q = False
        self.sync_freq = 200

        # For multi worker algorithms
        self.nworkers = 8

        # For n-step algorithms
        self.nsteps = 5

        # For actor-critic algorithms
        self.entropy_weight = 0.01
        self.value_loss_weight = 1.0
        self.use_gae = False
        self.gae_tau = 1.0

        # For acktr and ppo
        self.lr_decay = False

        # For ppo
        self.adv_normalize_eps = 1.0e-5
        self.ppo_minibatch_size = 32
        self.ppo_epochs = 10
        self.ppo_clip = 0.2
        self.ppo_value_clip = True
        self.clip_decay = False

        # Logger and logging frequency
        self.logger = Logger()
        self.episode_log_freq = 100
        self.network_log_freq = 1000
        self.eval_freq = 1000
        self.use_reward_monitor = False
        self.save_freq = 10000
        self.save_eval_actions = False

        self.__env = lambda: ClassicalControl()
        self.__eval_env: Optional[EnvExt] = None
        self.__exp: Callable[[ValuePredictor], Explorer] = \
            lambda net: EpsGreedy(1.0, LinearCooler(1.0, 0.1, 10000), net)
        self.__optim = lambda params: RMSprop(params, 0.001)
        self.__replay: Callable[[int], ReplayBuffer] = \
            lambda capacity: UniformReplayBuffer(DqnReplayFeed, capacity=capacity)
        self.__net: Dict[str, NetFn] = {
            'value': value.fc,
            'actor-critic': actor_critic.fc,
        }
        self.__paralle_env = lambda env_gen, num_w: DummyParallelEnv(env_gen, num_w)

    def env(self) -> EnvExt:
        env = self.__env()
        if self.state_dim == (0,):
            self.action_dim = env.action_dim
            self.state_dim = env.state_dim
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
        self.state_dim = env.state_dim
        self.__eval_env = env

    def explorer(self, value_pred: ValuePredictor) -> Explorer:
        return self.__exp(value_pred)

    def set_explorer(self, exp: Callable[[ValuePredictor], Explorer]) -> None:
        self.__exp = exp

    def optimizer(self, params: Params) -> Optimizer:
        return self.__optim(params)

    def set_optimizer(self, optim: Callable[[Params], Optimizer]) -> None:
        self.__optim = optim

    def replay_buffer(self) -> ReplayBuffer:
        return self.__replay(self.replay_size)

    def set_replay_buffer(self, replay: Callable[[int], ReplayBuffer]) -> None:
        self.__replay = replay

    def parallel_env(self) -> ParallelEnv:
        penv = self.__parallel_env(self.__env, self.nworkers)
        self.action_dim = penv.action_dim
        self.state_dim = penv.state_dim
        return penv

    def set_parallel_env(self, parallel_env: Callable[[Callable[[], EnvExt], int], ParallelEnv]):
        self.__parallel_env = parallel_env

    def net(self, name: str) -> nn.Module:
        assert self.state_dim[0] != 0
        assert self.action_dim != 0
        return self.__net[name](self.state_dim, self.action_dim, self.device)

    def set_net_fn(self, name: str, net: NetFn) -> None:
        self.__net[name] = net

    def lr_cooler(self, initial: float, minimum: float = 0.0) -> Cooler:
        if not self.lr_decay:
            return DummyCooler()
        return LinearCooler(initial, minimum, self.max_steps)

    def clip_cooler(self, minimum: float = 0.0) -> Cooler:
        if not self.clip_decay:
            return DummyCooler()
        return LinearCooler(self.ppo_clip, minimum, self.max_steps)


