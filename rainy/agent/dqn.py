from torch import nn
from .base import Agent
from ..config import Config
from ..net.value_net import ValueNet


class DqnAgent(Agent):
    def __init__(self, config: Config):
        self.config = config
        self.net = config.gen_value_net()
        self.target_net = config.gen_value_net()
        self.optimizer = config.gen_optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.policy = config.get_explorer(self.net)

        self.steps = 0

