from torch import nn, Tensor
from .base import Agent
from ..config import Config
from ..net.value_net import ValueNet

class DqnAgent(Agent):
    def __init__(self, config: Config):
        self.predict_net = config.value_net()
        self.target_net = config.value_net()
        self.optimizer = config.gen_optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.policy = config.get_explorer(self.net)
        self.steps = 0
        self.replay = config.replay_buffer()
        self.env = config.env()
        self.config = config

    def members_to_save(self):
        return "net", "target_net"

    def episode(self):
        pass

    def __episode(self, train: bool = False):
        total_reward = 0.0
        steps = 0
        self.env.seed(self.config.seed)
        state = self.config.wrap_state(self.env.reset())
        while True:
            action = self.policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if train:
                self.replay.append(state, action, reward, next_state, done)
                self.to


            


