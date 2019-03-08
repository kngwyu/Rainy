import numpy as np
import torch
from torch import nn
from typing import Optional, Tuple
from .base import NStepParallelAgent
from ..config import Config
from ..net import Policy
from ..prelude import Action, Array, State


class A2cAgent(NStepParallelAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('actor-critic')
        self.optimizer = config.optimizer(self.net.parameters())
        self.lr_cooler = config.lr_cooler(self.optimizer.param_groups[0]['lr'])

    def members_to_save(self) -> Tuple[str, ...]:
        return ("net",)

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    def eval_action(self, state: Array) -> Action:
        if len(state.shape) == len(self.net.state_dim):
            # treat as batch_size == 1
            state = np.stack([state])
        with torch.no_grad():
            policy = self.net.policy(state)
        if self.config.eval_deterministic:
            return policy.best_action().squeeze().cpu().numpy()
        else:
            return policy.action().squeeze().cpu().numpy()

    def eval_action_parallel(
            self,
            states: Array,
            ent: Optional[Array[float]] = None
    ) -> Array[Action]:
        with torch.no_grad():
            policy = self.net.policy(states)
        if ent is not None:
            ent += policy.entropy().cpu().numpy()
        if self.config.eval_deterministic:
            return policy.best_action().squeeze().cpu().numpy()
        else:
            return policy.action().squeeze().cpu().numpy()

    def _one_step(self, states: Array[State]) -> Array[State]:
        with torch.no_grad():
            policy, value, rnns = self.net(self.penv.states_to_array(states))
        next_states, rewards, done, info = self.penv.step(policy.action().squeeze().cpu().numpy())
        self.episode_length += 1
        self.rewards += rewards
        self.report_reward(done, info)
        self.storage.push(next_states, rewards, done, rnn_state=rnns, policy=policy, value=value)
        return next_states

    def _pre_backward(self, _policy: Policy, _value: torch.Tensor) -> None:
        pass

    def _step_optimizer(self) -> None:
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.lr_cooler.lr_decay(self.optimizer)

    def nstep(self, states: Array[State]) -> Array[State]:
        for _ in range(self.config.nsteps):
            states = self._one_step(states)
        with torch.no_grad():
            next_value = self.net.value(self.penv.states_to_array(states))
        if self.config.use_gae:
            gamma, tau = self.config.discount_factor, self.config.gae_tau
            self.storage.calc_gae_returns(next_value, gamma, tau)
        else:
            self.storage.calc_ac_returns(next_value, self.config.discount_factor)

        policy, value, _ = self.net(
            self.storage.batch_states(self.penv),
            self.storage.batch_rnn_states(self.net.recurrent_body)
        )
        policy.set_action(self.storage.batch_actions())

        advantage = self.storage.batch_returns() - value
        policy_loss = -(policy.log_prob() * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy_loss = policy.entropy().mean()
        self._pre_backward(policy, value)
        self.optimizer.zero_grad()
        (policy_loss
         + self.config.value_loss_weight * value_loss
         - self.config.entropy_weight * entropy_loss).backward()
        self._step_optimizer()
        self.report_loss(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
        )
        self.storage.reset()
        return states
