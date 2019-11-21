import torch
from torch import Tensor
from .a2c import A2CAgent
from ..lib.rollout import RolloutSampler
from ..lib import mpi
from ..config import Config
from ..envs import State
from ..net import ActorCriticNet, Policy
from ..prelude import Array


class PPOAgent(A2CAgent):
    SAVED_MEMBERS = "net", "clip_eps", "clip_cooler", "optimizer"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net: ActorCriticNet = config.net("actor-critic")  # type: ignore
        self.optimizer = config.optimizer(self.net.parameters())
        self.lr_cooler = config.lr_cooler(self.optimizer.param_groups[0]["lr"])
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        nbatchs = (
            self.config.nsteps * self.config.nworkers
        ) // self.config.ppo_minibatch_size
        self.num_updates = self.config.ppo_epochs * nbatchs
        mpi.setup_models(self.net)
        self.optimizer = mpi.setup_optimizer(self.optimizer)

    def _policy_loss(
        self, policy: Policy, advantages: Tensor, old_log_probs: Tensor
    ) -> Tensor:
        prob_ratio = torch.exp(policy.log_prob() - old_log_probs)
        surr1 = prob_ratio * advantages
        surr2 = prob_ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        return -torch.min(surr1, surr2).mean()

    def _value_loss(self, value: Tensor, old_value: Tensor, returns: Tensor) -> Tensor:
        """Clip value function loss.
        OpenAI baselines says it reduces variability during Critic training...
        but I'm not sure.
        """
        unclipped_loss = (value - returns).pow(2)
        if not self.config.ppo_value_clip:
            return unclipped_loss.mean()
        value_clipped = old_value + (value - old_value).clamp(
            -self.clip_eps, self.clip_eps
        )
        clipped_loss = (value_clipped - returns).pow(2)
        return torch.max(unclipped_loss, clipped_loss).mean()

    def nstep(self, states: Array[State]) -> Array[State]:
        for _ in range(self.config.nsteps):
            states = self._one_step(states)

        with torch.no_grad():
            next_value = self.net.value(*self._network_in(states))

        if self.config.use_gae:
            self.storage.calc_gae_returns(
                next_value, self.config.discount_factor, self.config.gae_lambda,
            )
        else:
            self.storage.calc_ac_returns(next_value, self.config.discount_factor)
        p, v, e = 0.0, 0.0, 0.0
        sampler = RolloutSampler(
            self.storage,
            self.penv,
            self.config.ppo_minibatch_size,
            rnn=self.net.recurrent_body,
            adv_normalize_eps=self.config.adv_normalize_eps,
        )
        for _ in range(self.config.ppo_epochs):
            for batch in sampler:
                policy, value, _ = self.net(batch.states, batch.rnn_init, batch.masks)
                policy.set_action(batch.actions)
                policy_loss = self._policy_loss(
                    policy, batch.advantages, batch.old_log_probs
                )
                value_loss = self._value_loss(value, batch.values, batch.returns)
                entropy_loss = policy.entropy().mean()
                self.optimizer.zero_grad()
                (
                    policy_loss
                    + self.config.value_loss_weight * 0.5 * value_loss
                    - self.config.entropy_weight * entropy_loss
                ).backward()
                mpi.clip_and_step(
                    self.net.parameters(), self.config.grad_clip, self.optimizer
                )
                p, v, e = (
                    p + policy_loss.item(),
                    v + value_loss.item(),
                    e + entropy_loss.item(),
                )

        self.lr_cooler.lr_decay(self.optimizer)
        self.clip_eps = self.clip_cooler()
        self.storage.reset()

        p, v, e = (x / self.num_updates for x in (p, v, e))
        self.network_log(policy_loss=p, value_loss=v, entropy_loss=e)
        return states
