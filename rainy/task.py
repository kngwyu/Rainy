from gym import Env


class Task(Env):
    """ Wrapper of gym.Env
    """
    def __init__(self, env: Env, action_dim: int, state_dim: int) -> None:
        self.env = env
        self.action_dim = action_dim
        self.state_dim = state_dim

    def step(self, action):
        observation, reward, done, info = self.env.step()
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()

    def render(self, mode='human'):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self, seed):
        return self.env.seed(seed)

    @property
    def unwrapped(self):
        return self.env.unwrapped
