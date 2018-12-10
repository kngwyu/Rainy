from functools import partial
import numpy as np
from numpy import ndarray
import torch
from torch import nn, Tensor
from typing import Iterable, List, Tuple
from .base import NStepAgent
from ..config import Config
from ..envs import Action, State


class A2cAgent(NStepAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.penv = config.parallel_env()
        self.net = config.net('actor-critic')
        self.optimizer = config.optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.rewards = np.zeros(config.num_workers, dtype=np.float32)

    def members_to_save(self) -> Tuple[str, ...]:
        return ("net",)

    def eval_action(self, state: State) -> Action:
        if self.config.eval_deterministic:
            return self.net.best_action(state).detach().item()
        else:
            return self.net(state)[0].detach().item()

    def _one_step(self, states: Iterable[State], episodic_rewards: list) -> Tuple[ndarray, ndarray]:
        action, log_prob, entropy, value = self.net(self.penv.states_to_array(states))
        next_states, rewards, is_term, _ = map(np.asarray, zip(*self.penv.step(action.detach())))
        self.rewards += rewards
        for i in range(self.config.num_workers):
            if is_term[i]:
                episodic_rewards.append(self.rewards[i])
                self.rewards[i] = 0
        rewards, is_term = map(lambda x: self.config.device.tensor(x), (rewards, 1.0 - is_term))
        return next_states, (action, log_prob, entropy, value, rewards, is_term)

    def _sum_up(self, reward_sum: Tensor, rollouts: ndarray) -> List[tuple]:
        res = [None] * self.config.nstep
        advantage = self.config.device.tensor(np.zeros(self.config.num_workers))
        for i in reversed(range(self.config.nstep)):
            action, log_prob, entropy, value, reward, is_term = rollouts[i]
            reward_sum = reward + reward_sum * self.config.discount_factor * is_term
            if not self.config.use_gae:
                advantage = reward_sum - value.detach()
            else:
                next_value = rollouts[i + 1][3]
                td_error = reward + self.config.discount_factor * is_term * next_value - value
                advantage *= \
                    self.config.gae_tau * self.config.discount_factor * is_term + td_error
            res[i] = (log_prob, entropy, value, reward_sum.detach(), advantage.detach())
        return res

    def nstep(self, states: Iterable[State]) -> Tuple[Iterable[State], Iterable[float]]:
        rollouts, episodic_rewards = [], []
        for _ in range(self.config.nstep):
            states, rollout = self._one_step(states, episodic_rewards)
            rollouts.append(rollout)
        next_value = self.net(self.penv.states_to_array(states))[3]
        rollouts.append((None, None, None, next_value, None, None))
        log_prob, entropy, value, reward_sum, advantage = \
            map(partial(torch.cat, dim=0), zip(*self._sum_up(next_value.detach(), rollouts)))
        policy_loss = -(log_prob * advantage).mean()
        value_loss = 0.5 * (reward_sum - value).pow(2).mean()
        entropy_loss = entropy.mean()
        self.optimizer.zero_grad()
        (policy_loss + self.config.value_loss_weight * value_loss -
         self.config.entropy_weight * entropy_loss).backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return states, episodic_rewards
