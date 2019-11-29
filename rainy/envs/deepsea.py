import gym
from gym.utils import seeding
import numpy as np
from typing import List, Optional, Tuple, Union


class DeepSea(gym.Env):
    SCREEN_SIZE = 400

    def __init__(self, size: int, noise: float = 0.0) -> None:
        self._size = size
        self._move_cost = 0.01 / size
        self._goal_reward = 1.0
        self._column = 0
        self._row = 0
        self.action_space = gym.spaces.Discrete(2)
        low = np.zeros(size ** 2)
        high = np.ones(size ** 2)
        self.observation_space = gym.spaces.Box(low, high)
        self.np_random = None
        self.noise = noise
        self._viewer = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # Remap actions according to column (action_right = go right)
        if self.noise == 0.0 or self.noise < self.np_random.uniform(0, 1):
            action_right = action == 1
        else:
            action_right = action != 1

        # Compute the reward
        reward = 0.0
        if self._column == self._size - 1 and action_right:
            reward += self._goal_reward

        # State dynamics
        if action_right:  # right
            self._column = np.clip(self._column + 1, 0, self._size - 1)
            reward -= self._move_cost
        else:  # left
            self._column = np.clip(self._column - 1, 0, self._size - 1)

        # Compute the observation
        self._row += 1
        if self._row == self._size:
            observation = self._get_observation(self._row - 1, self._column)
            return observation, reward, True, {}
        else:
            observation = self._get_observation(self._row, self._column)
            return observation, reward, False, {}

    def reset(self) -> np.ndarray:
        self._reset_next_step = False
        self._column = 0
        self._row = 0
        return self._get_observation(self._row, self._column)

    def render(self, mode: str = "human") -> Union[np.ndarray, bool]:
        player_size = self.SCREEN_SIZE / self._size
        if self._viewer is None:
            from gym.envs.classic_control import rendering

            self._viewer = rendering.Viewer(self.SCREEN_SIZE, self.SCREEN_SIZE)
            self.player_trans = rendering.Transform()
            v = np.array([(0.0, 0.0), (2.0, 0.0), (1.5, -1.0), (0.5, -1.0)])
            player = rendering.make_polygon(v * player_size / 2)
            player.set_color(0.0, 0.0, 1.0)
            player.add_attr(self.player_trans)
            self._viewer.add_geom(player)
        x = player_size * self._column
        y = self.SCREEN_SIZE - player_size * self._row
        self.player_trans.set_translation(x, y)
        return self._viewer.render(return_rgb_array=mode == "rgb_array")

    def _get_observation(self, row: int, column: int) -> np.ndarray:
        observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        observation[row, column] = 1
        return observation.flatten()
