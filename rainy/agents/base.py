import copy
import warnings

from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import click
import numpy as np
import torch
from torch import Tensor, nn

from ..config import Config
from ..envs import ParallelEnv
from ..lib import mpi
from ..net import DummyRnn, RnnState
from ..prelude import Action, Array, State, DEFAULT_SAVEFILE_NAME
from ..replay import ReplayFeed

Netout = Dict[str, Tensor]


class EpisodeResult(NamedTuple):
    return_: np.float32
    length: np.int32

    def __repr__(self) -> str:
        return f"EpisodeResult(Return: {self.return_}, Length: {self.length})"


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
    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
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

    def initialize_rollouts(self) -> None:
        """ This function must be called when train re-started in a different env.
        """
        self.storage.initialize()

    def close(self) -> None:
        self.env.close()
        if not self.config.keep_logger:
            self.logger.close()

    def __eval_episode(
        self,
        select_action: Callable[[Array, Optional[Netout]], Action],
        render: bool,
        pause: bool,
    ) -> EpisodeResult:
        return_ = 0.0
        steps = 0
        env = self.config.eval_env
        if self.config.seed is not None:
            env.seed(self.config.seed)
        state = env.reset()
        for hook in self.config.eval_hooks:
            hook.reset(self, env, state)
        if render:
            env.render()
            if pause:
                click.pause()
        while True:
            state = env.extract(state)
            net_out = {}
            action = select_action(state, net_out)
            transition = env.step_and_render(action, render)
            for hook in self.config.eval_hooks:
                hook.step(env, action, transition, net_out)
            steps += 1
            return_ += transition.reward
            state = transition.state
            res = self._result(transition.terminal, transition.info, return_, steps)
            if res is not None:
                self.eval_reset()
                break
        for hook in self.config.eval_hooks:
            hook.close()
        return res

    def _result(
        self, done: bool, info: dict, return_: float, episode_length: int,
    ) -> Optional[EpisodeResult]:
        if self.env.use_reward_monitor:
            if "episode" in info:
                return EpisodeResult(info["episode"]["r"], info["episode"]["l"])
        elif done:
            return EpisodeResult(return_, episode_length)
        return None

    def random_episode(
        self, render: bool = False, pause: bool = False
    ) -> EpisodeResult:
        def act(_state, *args) -> Action:
            return self.config.eval_env._spec.random_action()

        return self.__eval_episode(act, render, pause)

    def random_and_save(
        self, fname: str, render: bool = False, pause: bool = False
    ) -> EpisodeResult:
        def act(_state, *args) -> Action:
            return self.config.eval_env._spec.random_action()

        res = self.__eval_episode(act, render, pause)
        self.config.eval_env.save_history(fname)
        return res

    def eval_episode(self, render: bool = False, pause: bool = False) -> EpisodeResult:
        return self.__eval_episode(self.eval_action, render, pause)

    def eval_and_save(
        self, fname: str, render: bool = False, pause: bool = False
    ) -> EpisodeResult:
        res = self.__eval_episode(self.eval_action, render, pause)
        self.config.eval_env.save_history(fname)
        return res

    def savedict_hook(self, save_dict: dict) -> None:
        pass

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
        self.savedict_hook(save_dict)
        torch.save(save_dict, directory.joinpath(filename))

    def loaddict_hook(self, load_dict: dict) -> bool:
        pass

    def loadmember_hook(self, member_str: str, saved_item: dict) -> None:
        pass

    def load(self, path: Path = None) -> bool:
        if not mpi.IS_MPI_ROOT or not path.exists():
            return False
        if path.is_dir():
            path.joinpath(DEFAULT_SAVEFILE_NAME)
        saved_dict = torch.load(path, map_location=self.config.device.unwrapped)
        for member_str in self.SAVED_MEMBERS:
            if member_str in saved_dict:
                saved_item = saved_dict[member_str]
            else:
                warnings.warn("Member {} wasn't loaded".format(member_str))
                continue
            self.loadmember_hook(member_str, saved_item)
            mem = getattr(self, member_str)
            is_dataparallel = isinstance(mem, nn.DataParallel)
            if is_dataparallel or hasattr(mem, "state_dict"):
                try:
                    if is_dataparallel:
                        mem.module.load_state_dict(saved_item)
                    else:
                        mem.load_state_dict(saved_item)
                except Exception as e:
                    warnings.warn(f"Error ({e}) occured while loading {mem}")
            else:
                setattr(self, member_str, saved_item)
        self.loaddict_hook(saved_dict)
        return True

    def _backward(
        self,
        loss: Tensor,
        opt: torch.optim.Optimizer,
        params: Optional[Iterable[Tensor]] = None,
        grad_clip: Optional[float] = None,
    ) -> None:
        opt.zero_grad()
        loss.backward()
        grad_clip = grad_clip or self.config.grad_clip
        if params is not None and grad_clip is not None:
            nn.utils.clip_grad_norm_(params, grad_clip)
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
        terminal: bool,
    ) -> None:
        self.replay.append(state, action, next_state, reward, terminal)

    @property
    def train_started(self) -> bool:
        return self.config.train_start <= self.total_steps

    def batch_actions(self, states: Array[State], penv: ParallelEnv) -> Array[Action]:
        raise NotImplementedError(f"{type(self)} does not support action_parallel!")

    def train_episodes(self, max_steps: int) -> Iterable[List[EpisodeResult]]:
        if self.config.seed is not None:
            self.env.seed(self.config.seed)
        state = self.env.reset()
        return_, episode_length = 0.0, 0
        reward_scale = self.config.reward_scale
        while True:
            action = self.action(state)
            transition = self.env.step(action).map_r(lambda r: r * reward_scale)
            self.store_transition(state, action, *transition[:-1])  # Do not pass info
            if self.train_started and self.total_steps % self.config.update_freq == 0:
                self.train(self.replay.sample(self.config.replay_batch_size))
                self.update_steps += 1
            # Set next state
            state = transition.state
            # Update statistics
            self.total_steps += 1
            return_ += transition.reward
            episode_length += 1
            res = self._result(
                transition.terminal, transition.info, return_, episode_length
            )
            if res is not None:
                yield [res]
                state = self.env.reset()
                return_ = 0.0
                episode_length = 0
            if self.total_steps >= max_steps:
                break


class A2CLikeAgent(Agent, Generic[State]):
    """Agent with parallel env + nstep rollout
    """

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.returns = np.zeros(config.nworkers, dtype=np.float32)
        self.episode_length = np.zeros(config.nworkers, dtype=np.int32)
        self.episode_results: List[EpisodeResult] = []
        self.penv = config.parallel_env()
        self.eval_penv = None
        self.eval_rnns: RnnState = DummyRnn.DUMMY_STATE
        self.step_width = self.config.nsteps * self.config.nworkers * mpi.global_size()

        self.reward_scale = self.config.reward_scale

    def eval_parallel(self, n: Optional[int] = None) -> List[EpisodeResult]:
        reserved = (
            copy.deepcopy(self.returns),
            copy.deepcopy(self.episode_length),
            copy.deepcopy(self.episode_results),
        )
        self.returns.fill(0.0)
        self.episode_length.fill(0)
        self.episode_results.clear()
        n = n or self.config.nworkers

        if n < self.config.nworkers:
            diff = self.config.nworkers - n

            def _done(done):
                rest = np.zeros(diff, dtype=np.bool)
                return np.concatenate((done, rest))

        else:

            def _done(done):
                return done

        if self.eval_penv is None:
            self.eval_penv = self.config.parallel_env(min(n, self.config.nworkers))
            self.eval_penv.set_mode(train=False)
            self.config.set_parallel_seeds(self.penv)

        self.penv.copyto(self.eval_penv)
        states = self.eval_penv.reset()
        while True:
            actions = self.eval_action_parallel(self.penv.extract(states))
            states, rewards, done, info = self.eval_penv.step(actions)
            self.episode_length += 1
            self.returns[:n] += rewards
            self._report_reward(_done(done), info)
            if n <= len(self.episode_results):
                break

        res = self.episode_results
        self.returns, self.episode_length, self.episode_results = reserved
        self.eval_reset()
        return res

    @abstractmethod
    def eval_action_parallel(self, states: Array) -> Array[Action]:
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
                    EpisodeResult(self.returns[i], self.episode_length[i])
                )
        self.returns[done] = 0.0
        self.episode_length[done] = 0

    def _one_step(self, states: Array[State]) -> Array[State]:
        actions, net_outputs = self.actions(states)
        transition = self.penv.step(actions).map_r(lambda r: r * self.reward_scale)
        self.storage.push(*transition[:3], **net_outputs)
        self.returns += transition.rewards
        self.episode_length += 1
        self._report_reward(transition.terminals, transition.infos)
        return transition.states

    def initialize_rollouts(self) -> None:
        self.storage.initialize()

    def close(self) -> None:
        super().close()
        self.penv.close()
        if self.eval_penv is not None:
            self.eval_penv.close()

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

    def savedict_hook(self, save_dict: dict) -> None:
        nobs = self.penv.as_cls("NormalizeObsParallel")
        if nobs is not None:
            save_dict["obs_rms"] = nobs._rms
        nrew = self.penv.as_cls("NormalizeRewardParallel")
        if nrew is not None:
            save_dict["reward_rms"] = nrew._rms

    def loaddict_hook(self, load_dict: dict) -> None:
        if "obs_rms" in load_dict:
            nobs = self.penv.as_cls("NormalizeObsParallel")
            load_dict["obs_rms"].copyto(nobs._rms)
        elif "reward_rms" in load_dict:
            nrew = self.penv.as_cls("NormalizeRewardParallel")
            load_dict["reward_rms"].copyto(nrew._rms)
