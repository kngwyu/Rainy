from torch import nn
from torch.optim import Optimizer, RMSprop
from typing import Callable, Dict, List, Optional, Sequence
from .envs import ClassicControl, DummyParallelEnv, EnvExt, EnvGen, ParallelEnv
from .net import actor_critic, bootstrap, deterministic, option_critic, sac, value
from .net.prelude import NetFn
from .lib.explore import DummyCooler, Cooler, LinearCooler, Explorer, EpsGreedy
from .lib import mpi
from .lib.kfac import KfacPreConditioner, PreConditioner
from .prelude import Params
from .replay import DQNReplayFeed, ReplayBuffer, UniformReplayBuffer
from .utils import Device, ExperimentLogger


class Config:
    def __init__(self) -> None:
        # action/state dims are initialized lazily
        self.action_dim = 0
        self.state_dim: Sequence[int] = (0,)

        # Common parameters
        self.discount_factor = 0.99
        self.device = Device()
        self.grad_clip = 5.0  # I recommend 0.5 for A2C
        self.max_steps = 10000
        self.eval_deterministic = True

        # Replay buffer
        self.replay_batch_size = 64
        self.replay_size = 10000
        self.train_start = 1000
        self.__replay: Callable[
            [int], ReplayBuffer
        ] = lambda capacity: UniformReplayBuffer(DQNReplayFeed, capacity=capacity)

        # For the cases you can't set seed in constructor, like gym.atari
        self.seed: Optional[int] = None
        self.parallel_seeds: List[int] = []

        # For DQN-like algorithms
        self.sync_freq = 1000
        self.__explore: Dict[Optional[str], Callable[[], Explorer]] = {
            None: lambda: EpsGreedy(1.0, LinearCooler(1.0, 0.1, 10000)),
            "eval": lambda: EpsGreedy(0.01, DummyCooler(0.01)),
        }

        # For BootDQN
        self.num_ensembles = 10
        self.replay_prob = 0.5

        # Reward scaling
        # Currently only used by SAC
        self.reward_scale = 1.0

        # For algorithms that use soft updates(e.g., DDPG)
        self.soft_update_coef = 5e-3

        # For TD3
        self.policy_update_freq = 2

        # For SAC
        self.target_entropy = None
        self.automatic_entropy_tuning = True
        self.fixed_alpha = 1.0

        # For multi worker algorithms
        self.nworkers = 1

        # For n-step algorithms
        self.nsteps = 1

        # For actor-critic algorithms
        self.entropy_weight = 0.01
        self.value_loss_weight = 1.0
        self.use_gae = False
        self.gae_lambda = 1.0
        self.lr_min: Optional[float] = None  # Mujoco: None Atari 0.0

        # For ppo
        self.adv_normalize_eps = 1.0e-5
        self.ppo_minibatch_size = 64  # Mujoco: 64 Atari: 32 * 8
        self.ppo_epochs = 10  # Mujoco: 10 Atari: 3
        self.ppo_clip = 0.2  # Mujoco: 0.2 Atari: 0.1
        self.ppo_value_clip = True
        self.ppo_clip_min: Optional[float] = None  # Mujoco: None Atari: 0.0

        # For option critic
        self.opt_beta_adv_merginal = 0.0
        self.opt_delib_cost = 0.02

        # Logger and logging frequency
        self.logger = ExperimentLogger(mpi.IS_MPI_ROOT)
        self.eval_freq = 10000
        self.save_freq = int(1e6)
        self.save_eval_actions = False
        self.episode_log_freq = 1000
        self.network_log_freq = 10000

        # Optimizer and preconditioner
        self.__optim: Dict[Optional[str], Callable[[], Optimizer]] = {
            None: lambda params: RMSprop(params, 0.001)
        }
        self.__precond = lambda net: KfacPreConditioner(net)

        # Default Networks
        self.__net: Dict[str, NetFn] = {
            "dqn": value.fc(),
            "bootdqn": bootstrap.fc_separated(10),
            "actor-critic": actor_critic.fc_shared(),
            "ddpg": deterministic.fc_seprated(),
            "td3": deterministic.td3_fc_seprated(),
            "option-critic": option_critic.fc_shared(num_options=8),
            "sac": sac.fc_separated(),
        }

        # Environments
        self.eval_times = 1
        self.__env = lambda: ClassicControl()
        self.__eval_env: Optional[EnvExt] = None
        self.__paralle_env = lambda env_gen, num_w: DummyParallelEnv(env_gen, num_w)

    @property
    def batch_size(self) -> int:
        return self.nworkers * self.nsteps

    def env(self) -> EnvExt:
        env = self.__env()
        if self.state_dim == (0,):
            self.action_dim = env.action_dim
            self.state_dim = env.state_dim
        return env

    def set_env(self, env: EnvGen) -> None:
        self.__env = env

    @property
    def eval_env(self) -> EnvExt:
        if self.__eval_env is None:
            self.__eval_env = self.env()
        return self.__eval_env

    @eval_env.setter
    def eval_env(self, env: EnvExt) -> None:
        if self.state_dim == (0,):
            self.action_dim = env.action_dim
            self.state_dim = env.state_dim
        self.__eval_env = env

    def explorer(self, key: Optional[str] = None) -> Explorer:
        return self.__explore[key]()

    def set_explorer(
        self, exp: Callable[[], Explorer], key: Optional[str] = None
    ) -> None:
        self.__explore[key] = exp

    def optimizer(self, params: Params, key: Optional[str] = None) -> Optimizer:
        return self.__optim[key](params)

    def set_optimizer(
        self, optim: Callable[[Params], Optimizer], key: Optional[str] = None
    ) -> None:
        self.__optim[key] = optim

    def preconditioner(self, net: nn.Module) -> PreConditioner:
        return self.__precond(net)

    def set_preconditioner(
        self, precond: Callable[[nn.Module], PreConditioner]
    ) -> None:
        self.__precond = precond

    def replay_buffer(self) -> ReplayBuffer:
        return self.__replay(self.replay_size)

    def set_replay_buffer(self, replay: Callable[[int], ReplayBuffer]) -> None:
        self.__replay = replay

    def parallel_env(self) -> ParallelEnv:
        penv = self.__parallel_env(self.__env, self.nworkers)
        self.action_dim = penv.action_dim
        self.state_dim = penv.state_dim
        return penv

    def set_parallel_env(self, parallel_env: Callable[[EnvGen, int], ParallelEnv]):
        self.__parallel_env = parallel_env

    def set_parallel_seeds(self, penv: ParallelEnv) -> None:
        if len(self.parallel_seeds) == self.nworkers:
            penv.seed(self.parallel_seeds)
        elif self.seed is not None:
            penv.seed([self.seed] * self.nworkers)

    def net(self, name: str) -> nn.Module:
        assert self.state_dim[0] != 0
        assert self.action_dim != 0
        return self.__net[name](self.state_dim, self.action_dim, self.device)

    def set_net_fn(self, name: str, net: NetFn) -> None:
        self.__net[name] = net

    def _get_cooler(
        self, initial: float, minimal: Optional[float], update_span: int
    ) -> Cooler:
        if minimal is None:
            return DummyCooler(initial)
        return LinearCooler(initial, minimal, self.max_steps // update_span)

    def lr_cooler(self, initial: float) -> Cooler:
        return self._get_cooler(initial, self.lr_min, self.nsteps * self.nworkers)

    def clip_cooler(self) -> Cooler:
        return self._get_cooler(
            self.ppo_clip, self.ppo_clip_min, self.nsteps * self.nworkers
        )

    def __repr__(self) -> str:
        d = filter(lambda t: not t[0].startswith("_Config"), self.__dict__.items())
        return "Config: " + str(dict(d))
