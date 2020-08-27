"""
This module has an implementation of A2C, which is described in
- OpenAI Baselines: ACKTR & A2C
  - URL: https://openai.com/blog/baselines-acktr-a2c/

A2C is a synchronous version of A3C, which is described in
- http://proceedings.mlr.press/v48/mniha16.pdf
"""
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..config import Config
from ..lib.rollout import RolloutStorage
from ..net import ActorCriticNet, Policy, RnnState
from ..prelude import Action, Array, State
from .base import A2CLikeAgent, Netout


class A2CAgent(A2CLikeAgent[State]):
    SAVED_MEMBERS = "net", "lr_cooler"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.storage: RolloutStorage[State] = RolloutStorage(
            config.nsteps, config.nworkers, config.device
        )
        self.net: ActorCriticNet = config.net("actor-critic")  # type: ignore
        self.optimizer = config.optimizer(self.net.parameters())
        self.lr_cooler = config.lr_cooler(self.optimizer.param_groups[0]["lr"])
        self.eval_rnns = self.rnn_init()

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    def rnn_init(self) -> RnnState:
        return self.net.recurrent_body.initial_state(
            self.config.nworkers, self.config.device
        )

    def _reset(self, initial_states: Array[State]) -> None:
        self.storage.set_initial_state(initial_states, self.rnn_init())

    def eval_reset(self) -> None:
        self.eval_rnns.fill_(0.0)

    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        if state.ndim == len(self.net.state_dim):
            # treat as batch_size == 1
            state = np.stack([state])
        with torch.no_grad():
            policy, self.eval_rnns[0] = self.net.policy(
                state, self.eval_rnns[0].unsqueeze()
            )
        if net_outputs is not None:
            net_outputs["policy"] = policy
        return policy.eval_action(self.config.eval_deterministic)

    def eval_action_parallel(self, states: Array) -> Array[Action]:
        with torch.no_grad():
            policy, self.eval_rnns = self.net.policy(states, self.eval_rnns)
        return policy.eval_action(self.config.eval_deterministic)

    def _network_in(self, states: Array[State]) -> Tuple[Array, RnnState, torch.Tensor]:
        return (
            self.penv.extract(states),
            self.storage.rnn_states[-1],
            self.storage.masks[-1],
        )

    @torch.no_grad()
    def actions(self, states: Array[State]) -> Tuple[Array[Action], dict]:
        policy, value, rnns = self.net(*self._network_in(states))
        actions = policy.action().squeeze().cpu().numpy()
        return actions, dict(rnn_state=rnns, policy=policy, value=value)

    def _pre_backward(self, _policy: Policy, _value: torch.Tensor) -> None:
        pass

    def _step_optimizer(self) -> None:
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.lr_cooler.lr_decay(self.optimizer)

    def train(self, last_states: Array[State]) -> None:
        with torch.no_grad():
            next_value = self.net.value(*self._network_in(last_states))
        if self.config.use_gae:
            self.storage.calc_gae_returns(
                next_value, self.config.discount_factor, self.config.gae_lambda
            )
        else:
            self.storage.calc_ac_returns(next_value, self.config.discount_factor)
        policy, value, _ = self.net(
            self.storage.batch_states(self.penv),
            self.storage.rnn_states[0],
            self.storage.batch_masks(),
        )

        advantage = self.storage.returns[:-1].flatten() - value
        actions = self.storage.batch_actions()
        policy_loss = -(policy.log_prob(actions) * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy_loss = policy.entropy().mean()
        self._pre_backward(policy, value)
        self.optimizer.zero_grad()
        (
            policy_loss
            + self.config.value_loss_weight * 0.5 * value_loss
            - self.config.entropy_weight * entropy_loss
        ).backward()
        self._step_optimizer()
        self.network_log(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
        )
        self.storage.reset()
