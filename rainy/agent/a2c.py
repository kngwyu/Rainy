from functools import partial
import numpy as np
from numpy import ndarray
import torch
from torch import nn
from typing import List, Optional, Tuple
from .base import NStepAgent
from ..config import Config
from ..envs import Action, ParallelEnv, State


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

    def best_action(self, state: State) -> Action:
        return self.net(state)[0].detach().item()

    def nstep(self, states: List[State], nstep) -> Tuple[List[State], List[float]]:
        rollout = []
        episode_rewards = []
        for _ in range(nstep):
            action, log_prob, entropy, value = self.net(self.penv.states_to_array(states))
            next_states, rewards, is_term, _ = \
                map(np.asarray, zip(*self.penv.step(action.detach())))
            self.rewards += rewards
            for i in range(nstep):
                if is_term[i]:
                    episode_rewards.append(self.rewards[i])
                    self.rewards[i] = 0
            rewards, is_term = \
                map(partial(torch.tensor, dtype=torch.float32), (rewards, 1.0 - is_term))
            rollout.append((action, log_prob, entropy, value, rewards, is_term))
            states = next_states

        pending_value = self.net(self.penv.states_to_array(states))[-1].detach()
        rollout.append((None, pending_value, None, None, None, None))

        processed_rollout: List[Optional[tuple]] = [None] * nstep
        returns = pending_value
        for i in reversed(range(nstep)):
            action, log_prob, entropy, value, rewards, is_term = rollout[i]
            value = value.detach()
            next_value = rollout[i + 1][1].detach()
            returns = rewards + returns * self.config.discount_factor * is_term
            if not self.config.use_gae:
                advantange = returns - value.detach()
            else:
                td_error = rewards + self.config.discount_factor * is_term * next_value - value
                advantange *= \
                    self.config.gae_tau * self.config.discount_factor * is_term + td_error
            processed_rollout[i] = (log_prob, value, returns, advantange, entropy)

        log_prob, value, returns, advantage, entropy =\
            map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_prob * advantage
        value_loss = 0.5 * (returns - value).pow(2)
        entropy_loss = entropy.mean()

        self.optimizer.zero_grad()
        (policy_loss - self.config.entropy_weight * entropy_loss +
         self.config.value_loss_weight * value_loss).mean().backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return states, episode_rewards
