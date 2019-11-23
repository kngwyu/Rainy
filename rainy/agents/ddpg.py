from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple
from .base import OneStepAgent
from ..config import Config
from ..prelude import Action, Array, State


class DDPGAgent(OneStepAgent):
    SAVED_MEMBERS = "net", "target_net", "actor_opt", "critic_opt", "explorer", "replay"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net("ddpg")
        self.target_net = deepcopy(self.net)
        self.actor_opt = config.optimizer(self.net.actor_params(), key="actor")
        self.critic_opt = config.optimizer(self.net.critic_params(), key="critic")
        self.explorer = config.explorer()
        self.eval_explorer = config.explorer(key="eval")
        self.replay = config.replay_buffer()
        self.batch_indices = config.device.indices(config.replay_batch_size)

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

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
    def _q_next(self, next_states: Array) -> Tensor:
        actions = self.target_net.action(next_states)
        return self.target_net.q_value(next_states, actions)

    def _train(self) -> None:
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_array(self.env.extract) for ob in obs]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
        mask = self.config.device.tensor(1.0 - done)
        q_next = self._q_next(next_states)
        q_target = (
            q_next.squeeze_()
            .mul_(mask * self.config.discount_factor)
            .add_(self.config.device.tensor(rewards))
        )
        action, q_current = self.net(states, actions)

        #  Backward critic loss
        critic_loss = F.mse_loss(q_current.squeeze_(), q_target)
        self._backward(critic_loss, self.critic_opt, self.net.critic_params())

        #  Backward policy loss
        policy_loss = -self.net.q_value(states, action).mean()
        self._backward(policy_loss, self.actor_opt, self.net.actor_params())

        #  Update target network
        self.target_net.soft_update(self.net, self.config.soft_update_coef)
