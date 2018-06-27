from abc import ABC, abstractmethod
import gym

class Task(ABC):
    
    def reset(self):
        return self.env.reset()

    def step(self, action):
        next_state, reward, done, info = self.env.step()

    def seed(self, seed):
        return self.env.seed(seed)



