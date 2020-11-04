import os

import click
import numpy as np
from torch.optim import Adam

import rainy
from rainy.agents import PPOAgent


class PoBilliard(rainy.envs.EnvExt):
    def __init__(self, name: str) -> None:
        import gym
        import mujoco_maze  # noqa

        class CutObjectObs(gym.ObservationWrapper):
            def __init__(self, env: gym.Env) -> None:
                super().__init__(env)
                low = self.observation(self.observation_space.low)
                high = self.observation(self.observation_space.high)
                self.observation_space = gym.spaces.Box(low, high)

            def observation(self, obs: np.ndarray) -> np.ndarray:
                return np.concatenate((obs[:3], obs[6:]))

        super().__init__(CutObjectObs(gym.make(name)))
        self.action_shift = self.action_space.low
        self.action_scale = self.action_space.high - self.action_space.low

    def step(self, action: np.ndarray):
        action = self.action_scale / (1.0 + np.exp(-action)) + self.action_shift
        return super().step(action)


@rainy.main(PPOAgent, script_path=os.path.realpath(__file__))
def main(envname: str = "PointBilliard-v1", max_steps: int = int(1e6)) -> rainy.Config:
    c = rainy.Config()
    # Set the environment
    c.set_env(lambda: PoBilliard(envname))
    c.set_parallel_env(
        rainy.envs.pybullet_parallel(normalize_obs=False, normalize_reward=False),
    )
    # Set NN
    c.set_net_fn(
        "actor-critic",
        rainy.net.actor_critic.fc_shared(
            policy=rainy.net.policy.SeparateStdGaussianDist, rnn=rainy.net.GruBlock,
        ),
    )
    c.set_optimizer(lambda params: Adam(params, lr=3.0e-4, eps=1.0e-4))
    c.max_steps = max_steps
    c.grad_clip = 0.5
    # ppo params
    c.value_loss_weight = 0.5
    c.entropy_weight = 0.0
    c.gae_lambda = 0.95
    c.nworkers = 16
    c.nsteps = 128
    c.ppo_minibatch_size = (16 * 128) // 16
    c.ppo_clip = 0.2
    c.use_gae = True
    c.eval_freq = max_steps // 10
    c.eval_times = 6
    return c


if __name__ == "__main__":
    main()
