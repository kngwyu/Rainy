import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple
from .base import OneStepAgent
from ..config import Config
from ..prelude import Action, Array, State


class BootDQNAgent(OneStepAgent):
    SAVED_MEMBERS = "net", "policy", "total_steps"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not self.env.spec.is_discrete():
            raise RuntimeError("DQN only supports discrete action space.")
        self.net = config.net("bootdqn")
        self.optimizer = config.optimizer(self.net.parameters())
        self.policy = config.explorer()
        self.eval_policy = config.explorer(key="eval")
        self.batch_indices = config.device.indices(config.replay_batch_size)
        self.replays = []
        for _ in range(self.config.num_ensembles):
            replay = config.replay_buffer()
            replay.allow_overlap = True
            self.replays.append(replay)

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    @torch.no_grad()
    def eval_action(self, state: Array) -> Action:
        return self.eval_policy.select_action(state, self.net).item()  # type: ignore

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        action = self.policy.select_action(self.env.extract(state), self.net).item()
        next_state, reward, done, info = self.env.step(action)
        n_ens = self.config.num_ensembles
        mask = np.random.uniform(0, 1, n_ens) < self.config.replay_enqueue_prob
        for i in filter(lambda i: mask[i], range(n_ens)):
            self.replays[i].append(state, action, reward, next_state, done)
        if done:
            self._train()
            self.net.active_head = np.random.randint(n_ens)
        return next_state, reward, done, info

    @torch.no_grad()
    def _q_next(self, i: int, next_states: Array) -> Tensor:
        return self.net(i, next_states).max(axis=-1)[0]

    def _train(self) -> None:
        gamma = self.config.discount_factor
        loss = 0
        for i, replay in enumerate(self.replays):
            obs = replay.sample(self.config.replay_batch_size)
            obs = [ob.to_ndarray(self.env.extract) for ob in obs]
            states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
            q_next = self._q_next(i, next_states).mul_(self.tensor(1.0 - done))
            q_target = self.tensor(rewards).add_(q_next.mul_(gamma))
            q_current = self.net(i, states)[self.batch_indices, actions]
            loss += F.mse_loss(q_current, q_target)
        self._backward(loss, self.optimizer, self.net.parameters())
        self.network_log(q_value=q_current.mean().item(), value_loss=loss.item())
