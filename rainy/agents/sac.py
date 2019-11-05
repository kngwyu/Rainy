"""SAC(Soft Actor Critic) agent
Paper: https://arxiv.org/abs/1812.05905
"""
from abc import ABC, abstractmethod
import numpy as np
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from typing import Sequence, Tuple
from .base import OneStepAgent
from ..config import Config
from ..net import Policy
from ..prelude import Action, Array, State


class EntropyTuner(ABC):
    @abstractmethod
    def alpha(self, *args) -> float:
        pass


class DummyEntropyTuner(EntropyTuner):
    def __init__(self, alpha: float) -> None:
        self._alpha = alpha

    def alpha(self, *args) -> float:
        return self._alpha


class TrainableEntropyTuner(EntropyTuner):
    def __init__(self, target_entropy: float, config: Config) -> None:
        self.log_alpha = nn.Parameter(config.device.zeros(1))
        self.optim = config.optimizer([self.log_alpha], key='entropy')
        self.target_entropy = target_entropy

    def alpha(self, log_pi: Tensor, *args) -> float:
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
        self.optim.zero_grad()
        alpha_loss.backward()
        self.optim.step()
        return self.log_alpha.detach().exp().item()


class SacAgent(OneStepAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('sac')
        self.target_net = self.net.get_target()
        self.actor_opt = config.optimizer(self.net.actor_params(), key='actor')
        self.critic_opt = config.optimizer(self.net.critic_params(), key='critic')
        self.replay = config.replay_buffer()
        self.batch_indices = config.device.indices(config.replay_batch_size)
        if self.config.automatic_entropy_tuning:
            if self.config.target_entropy is None:
                target_entropy = 1.0
            else:
                target_entropy = -np.prod(self.env.state_dim).item() - 0.0
            self.entropy_tuner = TrainableEntropyTuner(target_entropy, config)
        else:
            self.entropy_tuner = DummyEntropyTuner(1.0)

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    def members_to_save(self) -> Sequence[str]:
        return 'net', 'target_net', 'actor_opt', 'critic_opt', 'total_steps', 'replay'

    @torch.no_grad()
    def eval_action(self, state: Array) -> Action:
        policy = self.net.policy(state)
        action = policy.best_action() if self.config.eval_deterministic else policy.action()
        return action.cpu().numpy()

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        train_started = self.total_steps > self.config.train_start
        if train_started:
            with torch.no_grad():
                policy = self.net.policy(state)
                action = policy.action().cpu().numpy()
        else:
            action = self.env.spec.random_action()
        action = self.env.spec.clip_action(action)
        next_state, reward, done, info = self.env.step(action)
        self.replay.append(state, action, reward * self.config.reward_scale, next_state, done)
        if train_started:
            self._train()
        return next_state, reward, done, info

    def _logpi_and_q(self, states: Tensor, policy: Policy) -> Tensor:
        actions = policy.baction()
        q1, q2 = self.net.q_values(states, actions)
        return policy.log_prob(use_baction=True), torch.min(q1, q2)

    @torch.no_grad()
    def _q_next(self, next_states: Tensor, alpha: Tensor) -> Tensor:
        policy = self.net.policy(next_states)
        q1, q2 = self.target_net.q_values(next_states, policy.action())
        return torch.min(q1, q2).squeeze_() - alpha * policy.log_prob()

    def _train(self) -> None:
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_ndarray(self.env.extract) for ob in obs]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
        q1, q2, policy = self.net(states, actions)

        # Backward policy loss
        log_pi, new_q = self._logpi_and_q(states, policy)
        alpha = self.entropy_tuner.alpha(log_pi)
        policy_loss = (alpha * log_pi - new_q).mean()
        self._backward(policy_loss, self.actor_opt, self.net.actor_params())

        #  Backward critic loss
        mask = self.config.device.tensor(1.0 - done)
        q_next = self._q_next(next_states, alpha)
        q_target = q_next.mul_(mask * self.config.discount_factor) \
                         .add_(self.config.device.tensor(rewards))
        critic_loss = F.mse_loss(q1.squeeze_(), q_target) + F.mse_loss(q2.squeeze_(), q_target)
        self._backward(critic_loss, self.critic_opt, self.net.critic_params())

        #  Update target network
        if (self.update_steps + 1) % self.config.sync_freq == 0:
            self.target_net.soft_update(self.net, self.config.soft_update_coef)
