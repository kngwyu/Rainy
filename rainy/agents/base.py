from abc import ABC, abstractmethod
import copy
import numpy as np
from pathlib import Path
import torch
from torch import nn
from typing import Callable, Generic, Iterable, List, NamedTuple, Optional, Tuple
import warnings
from ..config import Config
from ..lib import mpi
from ..net import DummyRnn, RnnState
from ..envs import EnvExt
from ..prelude import Action, Array, State
from ..utils.log import ExpStats


class EpisodeResult(NamedTuple):
    reward: np.float32
    length: np.int32

    def __repr__(self) -> str:
        return 'Result: reward: {}, length: {}'.format(self.reward, self.length)


class Agent(ABC):
    """Children must call super().__init__(config) first
    """
    def __init__(self, config: Config) -> None:
        self.config = config
        self.logger = config.logger
        self.env = config.env()
        self.total_steps = 0
        self.loss_stat = ExpStats()

    @abstractmethod
    def members_to_save(self) -> Tuple[str, ...]:
        """Here you can specify members you want to save.

    Examples::
        def members_to_save(self):
            return "net", "target"
        """
        pass

    @abstractmethod
    def train_episodes(self, max_steps: int) -> Iterable[List[EpisodeResult]]:
        """Train the agent.
        """
        pass

    @abstractmethod
    def eval_action(self, state: Array) -> Action:
        """Return the best action according to training results.
        """
        pass

    @property
    @abstractmethod
    def update_steps(self) -> int:
        pass

    def eval_reset(self) -> None:
        pass

    def set_mode(self, train: bool = True) -> None:
        pass

    def report_loss(self, **kwargs) -> None:
        if not mpi.IS_MPI_ROOT:
            return
        self.loss_stat.update(kwargs)
        if self.update_steps % self.config.loss_log_freq == 0:
            d = self.loss_stat.report_and_reset()
            d['update-steps'] = self.update_steps
            self.logger.exp('loss', d)

    def close(self) -> None:
        self.env.close()

    def __eval_episode(
            self,
            select_action: Callable[[Array], Action],
            render: bool
    ) -> Tuple[EpisodeResult, EnvExt]:
        total_reward = 0.0
        steps = 0
        env = self.config.eval_env
        if self.config.seed is not None:
            env.seed(self.config.seed)
        state = env.reset()
        while True:
            if render:
                env.render()
            state = env.extract(state)
            action = select_action(state)
            state, reward, done, info = env.step(action)
            steps += 1
            total_reward += reward
            res = self._result(done, info, total_reward, steps)
            if res is not None:
                self.eval_reset()
                return (res, env)

    def _result(
            self,
            done: bool,
            info: dict,
            total_reward: float,
            episode_length: int,
    ) -> Optional[EpisodeResult]:
        if self.env.use_reward_monitor:
            if 'episode' in info:
                return EpisodeResult(info['episode']['r'], info['episode']['l'])
        elif done:
            return EpisodeResult(total_reward, episode_length)
        return None

    def random_episode(self, render: bool = False) -> EpisodeResult:
        def act(_state) -> Action:
            return self.config.eval_env.spec.random_action()
        return self.__eval_episode(act, render)[0]

    def random_and_save(self, fname: str, render: bool = False) -> EpisodeResult:
        def act(_state) -> Action:
            return self.config.eval_env.spec.random_action()
        res, env = self.__eval_episode(act, render)
        env.save_history(fname)
        return res

    def eval_episode(self, render: bool = False) -> EpisodeResult:
        return self.__eval_episode(self.eval_action, render)[0]

    def eval_and_save(self, fname: str, render: bool = False) -> EpisodeResult:
        res, env = self.__eval_episode(self.eval_action, render)
        env.save_history(fname)
        return res

    def save(self, filename: str) -> None:
        if not mpi.IS_MPI_ROOT:
            return None
        save_dict = {}
        for member_str in self.members_to_save():
            value = getattr(self, member_str)
            if isinstance(value, nn.DataParallel):
                save_dict[member_str] = value.module.state_dict()
            elif hasattr(value, 'state_dict'):
                save_dict[member_str] = value.state_dict()
            else:
                save_dict[member_str] = value
        log_dir = self.config.logger.log_dir
        if log_dir is None:
            log_dir = Path('.')
        torch.save(save_dict, log_dir.joinpath(filename))

    def load(self, filename: str) -> None:
        if not mpi.IS_MPI_ROOT:
            return None
        saved_dict = torch.load(filename, map_location=self.config.device.unwrapped)
        #  For backward compatibility, we need to check both index and name
        for idx, member_str in enumerate(self.members_to_save()):
            if idx in saved_dict:
                saved_item = saved_dict[idx]
            elif member_str in saved_dict:
                saved_item = saved_dict[member_str]
            else:
                warnings.warn('Member {} wasn\'t loaded'.format(member_str))
                continue
            mem = getattr(self, member_str)
            if isinstance(mem, nn.DataParallel):
                mem.module.load_state_dict(saved_item)
            elif hasattr(mem, 'state_dict'):
                mem.load_state_dict(saved_item)
            else:
                setattr(self, member_str, saved_item)


class OneStepAgent(Agent, Generic[State]):
    @abstractmethod
    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        pass

    @property
    def update_steps(self) -> int:
        return self.total_steps

    def train_episodes(self, max_steps: int) -> Iterable[List[EpisodeResult]]:
        if self.config.seed is not None:
            self.env.seed(self.config.seed)
        state = self.env.reset()
        total_reward = 0.0
        episode_length = 0
        while True:
            state, reward, done, info = self.step(state)
            self.total_steps += 1
            total_reward += reward
            episode_length += 1
            res = self._result(done, info, total_reward, episode_length)
            if res is not None:
                yield [res]
                state = self.env.reset()
                total_reward = 0.0
                episode_length = 0
            if self.total_steps >= max_steps:
                break


class NStepParallelAgent(Agent, Generic[State]):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.rewards = np.zeros(config.nworkers, dtype=np.float32)
        self.episode_length = np.zeros(config.nworkers, dtype=np.int)
        self.episode_results: List[EpisodeResult] = []
        self.penv = config.parallel_env()
        self.eval_rnns: RnnState = DummyRnn.DUMMY_STATE
        self.step_width = self.config.nsteps * self.config.nworkers * mpi.global_size()

    def eval_parallel(
            self,
            n: Optional[int] = None,
            entropy: Optional[Array[float]] = None,
    ) -> List[EpisodeResult]:
        reserved = (
            copy.deepcopy(self.rewards),
            copy.deepcopy(self.episode_length),
            copy.deepcopy(self.episode_results),
        )
        self.rewards.fill(0.0)
        self.episode_length.fill(0)
        self.episode_results.clear()
        if n is None:
            n = self.config.nworkers
        self.config.set_parallel_seeds(self.penv)

        states = self.penv.reset()
        mask = self.config.device.ones(self.config.nworkers)
        while True:
            actions = self.eval_action_parallel(self.penv.extract(states), mask, entropy)
            states, rewards, done, info = self.penv.step(actions)
            self.episode_length += 1
            self.rewards += rewards
            self.report_reward(done, info)
            if n <= len(self.episode_results):
                break

        res = self.episode_results
        self.rewards, self.episode_length, self.episode_results = reserved
        self.eval_reset()
        return res

    @abstractmethod
    def eval_action_parallel(
            self,
            states: Array,
            mask: torch.Tensor,
            ent: Optional[Array[float]] = None,
    ) -> Array[Action]:
        pass

    @abstractmethod
    def nstep(self, states: Array[State]) -> Array[State]:
        pass

    @abstractmethod
    def _reset(self, initial_states: Array[State]) -> None:
        pass

    @property
    def update_steps(self) -> int:
        return self.total_steps // self.step_width

    def rnn_init(self) -> RnnState:
        return DummyRnn.DUMMY_STATE

    def report_reward(self, done: Array[bool], info: Array[dict]) -> None:
        if self.penv.use_reward_monitor:
            for i in filter(lambda i: 'episode' in i, info):
                self.episode_results.append(EpisodeResult(i['episode']['r'], i['episode']['l']))
        else:
            for i in filter(lambda i: done[i], range(self.config.nworkers)):  # type: ignore
                self.episode_results.append(EpisodeResult(self.rewards[i], self.episode_length[i]))
                self.rewards[i] = 0.0
                self.episode_length[i] = 0

    def close(self) -> None:
        self.env.close()
        self.penv.close()

    def train_episodes(self, max_steps: int) -> Iterable[List[EpisodeResult]]:
        self.config.set_parallel_seeds(self.penv)
        states = self.penv.reset()
        self._reset(states)
        while True:
            states = self.nstep(states)
            self.total_steps += self.step_width
            if self.episode_results:
                yield self.episode_results
                self.episode_results = []
            if self.total_steps >= max_steps:
                break
