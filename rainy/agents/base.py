from abc import ABC, abstractmethod
import click
import copy
import numpy as np
from pathlib import Path
import torch
from torch import nn, Tensor
from typing import (
    Callable,
    ClassVar,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)
import warnings
from ..config import Config
from ..lib import mpi
from ..net import DummyRnn, RnnState
from ..envs import EnvExt, ParallelEnv
from ..prelude import Action, Array, State
from ..replay import ReplayFeed


class EpisodeResult(NamedTuple):
    reward: np.float32
    length: np.int32

    def __repr__(self) -> str:
        return f"EpisodeResult(reward: {self.reward}, length: {self.length})"


class Agent(ABC):
    """Children must call super().__init__(config) first
    """

    SAVED_MEMBERS: ClassVar[Sequence[str]]
    update_steps: int

    def __init__(self, config: Config) -> None:
        self.config = config
        self.tensor = config.device.tensor
        self.logger = config.logger
        self.env = config.env()
        self.total_steps = 0
        self.logger.summary_setting(
            "network",
            ["total_steps", "update_steps"],
            interval=config.network_log_freq,
            color="blue",
        )

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

    def eval_reset(self) -> None:
        pass

    def set_mode(self, train: bool = True) -> None:
        pass

    def network_log(self, **kwargs) -> None:
        kwargs["total_steps"] = self.total_steps
        kwargs["update_steps"] = self.update_steps
        self.logger.submit("network", **kwargs)

    def close(self) -> None:
        self.env.close()
        self.logger.close()

    def __eval_episode(
        self, select_action: Callable[[Array], Action], render: bool, pause: bool
    ) -> Tuple[EpisodeResult, EnvExt]:
        total_reward = 0.0
        steps = 0
        env = self.config.eval_env
        if self.config.seed is not None:
            env.seed(self.config.seed)
        state = env.reset()
        if render:
            env.render()
            if pause:
                click.pause()
        while True:
            state = env.extract(state)
            action = select_action(state)
            state, reward, done, info = env.step_and_render(action, render)
            steps += 1
            total_reward += reward
            res = self._result(done, info, total_reward, steps)
            if res is not None:
                self.eval_reset()
                return (res, env)

    def _result(
        self, done: bool, info: dict, total_reward: float, episode_length: int,
    ) -> Optional[EpisodeResult]:
        if self.env.use_reward_monitor:
            if "episode" in info:
                return EpisodeResult(info["episode"]["r"], info["episode"]["l"])
        elif done:
            return EpisodeResult(total_reward, episode_length)
        return None

    def random_episode(
        self, render: bool = False, pause: bool = False
    ) -> EpisodeResult:
        def act(_state) -> Action:
            return self.config.eval_env.spec.random_action()

        return self.__eval_episode(act, render, pause)[0]

    def random_and_save(
        self, fname: str, render: bool = False, pause: bool = False
    ) -> EpisodeResult:
        def act(_state) -> Action:
            return self.config.eval_env.spec.random_action()

        res, env = self.__eval_episode(act, render, pause)
        env.save_history(fname)
        return res

    def eval_episode(self, render: bool = False, pause: bool = False) -> EpisodeResult:
        return self.__eval_episode(self.eval_action, render, pause)[0]

    def eval_and_save(
        self, fname: str, render: bool = False, pause: bool = False
    ) -> EpisodeResult:
        res, env = self.__eval_episode(self.eval_action, render, pause)
        env.save_history(fname)
        return res

    def save(self, filename: str, directory: Optional[Path] = None) -> None:
        if not mpi.IS_MPI_ROOT:
            return None
        save_dict = {}
        for member_str in self.SAVED_MEMBERS:
            value = getattr(self, member_str)
            if isinstance(value, nn.DataParallel):
                save_dict[member_str] = value.module.state_dict()
            elif hasattr(value, "state_dict"):
                save_dict[member_str] = value.state_dict()
            else:
                save_dict[member_str] = value
        if directory is None:
            directory = self.logger.logdir
        torch.save(save_dict, directory.joinpath(filename))

    def load(self, filename: str, directory: Optional[Path] = None) -> bool:
        if not mpi.IS_MPI_ROOT:
            return False
        if directory is None:
            directory = self.logger.logdir
        path = directory.joinpath(filename)
        if not path.exists():
            return False
        saved_dict = torch.load(path, map_location=self.config.device.unwrapped)
        #  For backward compatibility, we need to check both index and name
        for idx, member_str in enumerate(self.SAVED_MEMBERS):
            if idx in saved_dict:
                saved_item = saved_dict[idx]
            elif member_str in saved_dict:
                saved_item = saved_dict[member_str]
            else:
                warnings.warn("Member {} wasn't loaded".format(member_str))
                continue
            mem = getattr(self, member_str)
            if isinstance(mem, nn.DataParallel):
                mem.module.load_state_dict(saved_item)
            elif hasattr(mem, "state_dict"):
                mem.load_state_dict(saved_item)
            else:
                setattr(self, member_str, saved_item)
        return True

    def _backward(
        self,
        loss: Tensor,
        opt: torch.optim.Optimizer,
        params: Optional[Iterable[Tensor]] = None,
    ) -> None:
        opt.zero_grad()
        loss.backward()
        if params is not None and self.config.grad_clip is not None:
            nn.utils.clip_grad_norm_(params, self.config.grad_clip)
        opt.step()


class DQNLikeAgent(Agent, Generic[State, Action, ReplayFeed]):
    """Agent with 1 step rollout + replay buffer
    """

    SUPPORT_PARALLEL_ENV = False

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.update_steps = 1

    @abstractmethod
    def action(self, state: State) -> Action:
        pass

    @abstractmethod
    def train(self, replay_feed: ReplayFeed) -> None:
        pass

    def store_transition(
        self,
        state: State,
        action: Action,
        next_state: State,
        reward: float,
        done: bool,
    ) -> None:
        reward *= self.config.reward_scale
        self.replay.append(state, action, next_state, reward, done)

    @property
    def train_started(self) -> bool:
        return self.config.train_start <= self.total_steps

    def batch_actions(self, states: Array[State], penv: ParallelEnv) -> Array[Action]:
        raise NotImplementedError(f"{type(self)} does not support action_parallel!")

    def train_episodes(self, max_steps: int) -> Iterable[List[EpisodeResult]]:
        if self.config.seed is not None:
            self.env.seed(self.config.seed)
        state = self.env.reset()
        total_reward = 0.0
        episode_length = 0
        while True:
            action = self.action(state)
            next_state, reward, done, info = self.env.step(action)
            self.store_transition(state, action, next_state, reward, done)
            if self.train_started and self.total_steps % self.config.update_freq == 0:
                self.train(self.replay.sample(self.config.replay_batch_size))
                self.update_steps += 1
            # Set next state
            state = next_state
            # Update stats
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


class A2CLikeAgent(Agent, Generic[State]):
    """Agent with parallel env + nstep rollout
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.rewards = np.zeros(config.nworkers, dtype=np.float32)
        self.episode_length = np.zeros(config.nworkers, dtype=np.int)
        self.episode_results: List[EpisodeResult] = []
        self.penv = config.parallel_env()
        self.eval_penv = None
        self.eval_rnns: RnnState = DummyRnn.DUMMY_STATE
        self.step_width = self.config.nsteps * self.config.nworkers * mpi.global_size()

    def eval_parallel(
        self, n: Optional[int] = None, entropy: Optional[Array[float]] = None,
    ) -> List[EpisodeResult]:
        reserved = (
            copy.deepcopy(self.rewards),
            copy.deepcopy(self.episode_length),
            copy.deepcopy(self.episode_results),
        )
        self.rewards.fill(0.0)
        self.episode_length.fill(0)
        self.episode_results.clear()
        n = n or self.config.nworkers

        if self.eval_penv is None:
            self.eval_penv = self.config.parallel_env(min(n, self.config.nworkers))
            self.config.set_parallel_seeds(self.penv)

        states = self.eval_penv.reset()
        mask = self.config.device.ones(self.config.nworkers)
        while True:
            actions = self.eval_action_parallel(
                self.penv.extract(states), mask, entropy
            )
            states, rewards, done, info = self.eval_penv.step(actions)
            self.episode_length += 1
            self.rewards[:n] += rewards
            self._report_reward(done, info)
            if n <= len(self.episode_results):
                break

        res = self.episode_results
        self.rewards, self.episode_length, self.episode_results = reserved
        self.eval_reset()
        return res

    @abstractmethod
    def eval_action_parallel(
        self, states: Array, mask: torch.Tensor, ent: Optional[Array[float]] = None,
    ) -> Array[Action]:
        pass

    @abstractmethod
    def actions(self, states: Array[State]) -> Tuple[Array[Action], dict]:
        pass

    @abstractmethod
    def train(self, last_states: Array[State]) -> None:
        pass

    @abstractmethod
    def _reset(self, initial_states: Array[State]) -> None:
        pass

    @property
    def update_steps(self) -> int:
        return self.total_steps // self.step_width

    def rnn_init(self) -> RnnState:
        return DummyRnn.DUMMY_STATE

    def _report_reward(self, done: Array[bool], info: Array[dict]) -> None:
        if self.penv.use_reward_monitor:
            for i in filter(lambda i: "episode" in i, info):
                self.episode_results.append(
                    EpisodeResult(i["episode"]["r"], i["episode"]["l"])
                )
        else:
            for i in filter(lambda i: done[i], range(len(done))):
                self.episode_results.append(
                    EpisodeResult(self.rewards[i], self.episode_length[i])
                )
                self.rewards[i] = 0.0
                self.episode_length[i] = 0

    def _one_step(self, states: Array[State]) -> Array[State]:
        actions, net_outputs = self.actions(states)
        states, rewards, terminals, infos = self.penv.step(actions)
        self.episode_length += 1
        self.rewards += rewards
        self._report_reward(terminals, infos)
        self.storage.push(states, rewards, terminals, **net_outputs)
        return states

    def close(self) -> None:
        self.env.close()
        self.penv.close()
        self.logger.close()

    def train_episodes(self, max_steps: int) -> Iterable[List[EpisodeResult]]:
        self.config.set_parallel_seeds(self.penv)
        states = self.penv.reset()
        self._reset(states)
        while True:
            for _ in range(self.config.nsteps):
                states = self._one_step(states)
            self.train(states)
            self.total_steps += self.step_width
            if len(self.episode_results) > 0:
                yield self.episode_results
                self.episode_results = []
            if self.total_steps >= max_steps:
                break
