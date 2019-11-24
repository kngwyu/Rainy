from gym import spaces, logger
from gym.envs.classic_control import CartPoleEnv
import numpy as np

F32_MAX = np.finfo(np.float32).max


class CartPoleSwingUp(CartPoleEnv):
    START_POSITIONS = ["arbitary", "bottom"]

    def __init__(
        self,
        start_position="arbitary",
        height_threshold=0.5,
        theta_dot_threshold=1.0,
        x_reward_threshold=1.0,
    ):
        super().__init__()
        self.start_position = self.START_POSITIONS.index(start_position)
        high = np.array([self.x_threshold * 2, F32_MAX, np.pi, F32_MAX])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._height_threshold = height_threshold
        self._theta_dot_threshold = theta_dot_threshold
        self._x_reward_threshold = x_reward_threshold

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = x, x_dot, theta, theta_dot
        done = bool(x < -self.x_threshold or x > self.x_threshold)

        def _reward():
            is_upright = np.cos(theta) > self._height_threshold
            is_upright &= np.abs(theta_dot) < self._theta_dot_threshold
            is_upright &= np.abs(x) < self._x_reward_threshold
            return 1.0 if is_upright else 0.0

        if not done:
            reward = _reward()
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = _reward()
        else:
            if self.steps_beyond_done == 0:
                logger.warn("You are calling 'step()' after the episode ends.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(-0.05, 0.05, size=(4,))
        if self.start_position == 0:
            self.state[2] = self.np_random.uniform(-np.pi, np.pi)
        else:
            self.state[2] += np.pi
        self.steps_beyond_done = None
        return np.array(self.state)
