import numpy as np
from numpy import ndarray
import torch
from torch import nn
from typing import Iterable, List, Optional, Tuple
from .base import Agent
from ..config import Config
from ..env_ext import Action, ParallelEnv, State


class A2cAgent(Agent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.penv = config.parallel_env()
        self.net = config.actor_critic_net()
        self.optimizer = config.optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.episode_rewards: List[float] = []
        self.online_rewards = np.zeros(config.num_workers)

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net", "policy"

    def nstep(self, states: Iterable[State]) -> Iterable[State]:
        rollout = []
        for _ in range(self.config.nstep):
            actions, log_probs, entropys, values = self.net(self.penv.states_to_array(states))
            next_states, rewards, is_terms, _ = \
                map(np.asarray, zip(*self.penv.step(actions.detach().cpu())))
            self.online_rewards += rewards
            for i, is_term in enumerate(is_terms):
                if is_term:
                    self.episode_rewards.append(self.online_rewards)
                    self.online_rewards[i] = 0
            rollout.append((log_probs, values, actions, rewards, 1.0 - is_terms, entropys))
            states = next_states

        pending_value = self.net(self.penv.states_to_array(states))[-1].detach()
        rollout.append((None, pending_value, None, None, None, None))

        processed_rollout: List[Optional[tuple]] = [None for _ in range(self.config.nstep)]
        returns = pending_value
        for i in reversed(range(self.config.nstep)):
            log_probs, values, actions, rewards, is_terms, entropys = rollout[i]
            values = values.detach()
            next_values = rollout[i + 1][1].detach()
            returns = rewards + self.config.discount_factor * is_terms * returns
            if not self.config.use_gae:
                advantanges = returns - values.detach()
            else:
                tde = rewards + self.config.discount_factor * is_terms * next_values - values
                advantanges *= self.config.gae_tau * self.config.discount_factor * is_terms + tde
            processed_rollout[i] = (log_probs, values, returns, advantanges, entropys)

        log_probs, values, returns, advantages, entropys =\
            map(lambda x: torch.cat(x, dim=0), zip(*processed_rollout))
        policy_loss = -log_probs * advantages
        value_loss = 0.5 * (returns - values).pow(2)
        entropy_loss = entropys.mean()

        self.optimizer.zero_grad()
        (policy_loss - self.config.entropy_weight * entropy_loss +
         self.config.value_loss_weight * value_loss).mean().backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return states
