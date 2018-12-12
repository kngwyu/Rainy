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
        if len(state.shape) == len(self.net.state_dim):
            # treat as batch_size == 1
            state = np.stack([state])
        policy = self.net.policy(state)
        if self.config.eval_deterministic:
            return policy.best_action()
        else:
            return policy.action()

    def _one_step(self, states: Iterable[State], episodic_rewards: list) -> Tuple[ndarray, tuple]:
        policy, value = self.net(self.penv.states_to_array(states))
        next_states, rewards, done, _ = map(np.asarray, zip(*self.penv.step(policy.action())))
        self.rewards += rewards
        for i in range(self.config.num_workers):
            if done[i]:
                episodic_rewards.append(self.rewards[i])
                self.rewards[i] = 0
        rewards, done = map(lambda x: self.config.device.tensor(x), (rewards, 1.0 - done))
        return next_states, (policy, value, rewards, done)

    def _calc_returns(self, return_: Tensor, rollouts: ndarray) -> List[tuple]:
        res = [None] * self.config.nstep
        advantage = torch.zeros(self.config.num_workers, device=self.config.device())
        for i in reversed(range(self.config.nstep)):
            policy, value, reward, done = rollouts[i]
            mask = self.config.discount_factor * done
            return_ = reward + return_ * mask
            if not self.config.use_gae:
                advantage = return_ - value.detach()
            else:
                td_error = reward + rollouts[i + 1][1].detach() * mask - value.detach()
                advantage = advantage * self.config.gae_tau * mask + td_error
            res[i] = (policy.log_prob(), policy.entropy(),
                      value, return_.detach(), advantage.detach())
        return res

    def nstep(self, states: Iterable[State]) -> Tuple[Iterable[State], Iterable[float]]:
        rollouts, episodic_rewards = [], []
        for _ in range(self.config.nstep):
            states, rollout = self._one_step(states, episodic_rewards)
            rollouts.append(rollout)
        next_value = self.net(self.penv.states_to_array(states)).value
        rollouts.append((None, next_value, None, None))
        log_prob, entropy, value, return_, advantage = \
            map(partial(torch.cat, dim=0), zip(*self._calc_returns(next_value.detach(), rollouts)))
        policy_loss = -(log_prob * advantage).mean()
        value_loss = 0.5 * (return_ - value).pow(2).mean()
        entropy_loss = entropy.mean()
        self.optimizer.zero_grad()
        (policy_loss + self.config.value_loss_weight * value_loss -
         self.config.entropy_weight * entropy_loss).backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return states, episodic_rewards
