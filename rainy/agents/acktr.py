from .a2c import A2cAgent
from ..config import Config


class AcktrAgent(A2cAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.precond = config.preconditiner(self.net)

    def _step_optimizer(self) -> None:
        self.precond.step()
        self.optimizer.step()
