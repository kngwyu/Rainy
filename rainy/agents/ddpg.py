from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from typing import Tuple
from .base import OneStepAgent
from ..config import Config
from ..prelude import Action, Array, State


class DdpgAgent(OneStepAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('ddpg')
        self.target_net = deepcopy(self.net)
        self.actor_optim = config.optimizer(self.net.actor.parameters(), key='actor')
        self.critic_optim = config.optimizer(self.net.critic.parameters(), key='critic')
        self.explorer = config.explorer()
        self.replay = config.replay_buffer()
        self.batch_indices = config.device.indices(config.replay_batch_size)

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net", "policy", "total_steps"

    @torch.no_grad()
    def eval_action(self, state: Array) -> Action:
        return self.eval_policy.select_action(state, self.net).item()  # type: ignore

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        train_started = self.total_steps > self.config.train_start
        if train_started:
            action = self.explorer.add_noise(self.net.action(state))
        else:
            action = self.env.spec.random_action()
        action = self.env.spec.clip_action(action)
        next_state, reward, done, info = self.env.step(action)
        self.replay.append(state, action, reward, next_state, done)
        if train_started:
            self._train()
        return next_state, reward, done, info

    @torch.no_grad()
    def _next(self, next_states: Array) -> Tuple[Tensor, Tensor]:
        return self.target_net(next_states)

    def _train(self) -> None:
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_ndarray(self.env.extract) for ob in obs]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
        a_next, q_next = self._next(states, actions)
        q_next *= self.config.device.tensor(1.0 - done) * self.config.discount_factor
        q_next += self.config.device.tensor(rewards)
        q_current = self.net.q_value(states, actions)

        critic_loss = (q_current - q_next).pow(2).mul(0.5).sum(-1).mean()
        self.net.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action = self.net.action(states)
        policy_loss = -self.net.q_value(states.detach(), action).mean()

        self.net.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()

        self.target_net.soft_update(self.net, self.config.soft_update_coef)
