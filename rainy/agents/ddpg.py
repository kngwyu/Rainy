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
        self.actor_opt = config.optimizer(self.net.actor.parameters(), key='actor')
        self.critic_opt = config.optimizer(self.net.critic.parameters(), key='critic')
        self.explorer = config.explorer()
        self.eval_explorer = config.eval_explorer()
        self.replay = config.replay_buffer()
        self.batch_indices = config.device.indices(config.replay_batch_size)

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net", "policy", "total_steps"

    @torch.no_grad()
    def eval_action(self, state: Array) -> Action:
        action = self.eval_explorer.add_noise(self.net.action(state))
        return action.cpu().numpy()  # type: ignore

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        train_started = self.total_steps > self.config.train_start
        if train_started:
            with torch.no_grad():
                action = self.net.action(state)
            action = self.explorer.add_noise(action).cpu().numpy()
        else:
            action = self.env.spec.random_action()
        action = self.env.spec.clip_action(action)
        next_state, reward, done, info = self.env.step(action)
        self.replay.append(state, action, reward, next_state, done)
        if train_started:
            self._train()
        return next_state, reward, done, info

    @torch.no_grad()
    def _next(self, next_states: Array, next_actions: Array) -> Tuple[Tensor, Tensor]:
        return self.target_net(next_states, next_actions)

    def _train(self) -> None:
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_ndarray(self.env.extract) for ob in obs]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
        mask = self.config.device.tensor(1.0 - done)
        a_next, q_next = self._next(states, actions)
        q_next.squeeze_() \
              .mul_(mask * self.config.discount_factor) \
              .add_(self.config.device.tensor(rewards))
        q_current = self.net.q_value(states, actions).squeeze_()

        critic_loss = (q_current - q_next).pow(2).mul(0.5).mean()
        self.net.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        action = self.net.action(states)
        policy_loss = -self.net.q_value(states, action).mean()

        self.net.zero_grad()
        policy_loss.backward()
        self.actor_opt.step()
        self.target_net.soft_update(self.net, self.config.soft_update_coef)
