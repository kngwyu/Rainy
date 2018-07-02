from abc import ABC, abstractmethod
import gym

class Task(ABC):
    """ task
    """
    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step()

    def seed(self, seed):
        return self.env.seed(seed)

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass

    @abstractmethod
    def convert_action(self):
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass


class TaskGen():
    def __init__(self) -> None:
        pass

    def __call__(self) -> Task:
        pass
