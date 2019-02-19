from torch import Tensor
from torch.distributions import Categorical, Distribution
from .block import NetworkBlock


class Policy:
    def __init__(self, dist: Distribution) -> None:
        self.dist = dist
        self._action = None

    def detach_(self) -> None:
        self.dist.detach_()

    def action(self) -> Tensor:
        if self._action is None:
            self._action = self.dist.sample().detach()
        return self._action

    def best_action(self) -> Tensor:
        return self.dist.probs.argmax(dim=-1)

    def log_prob(self) -> Tensor:
        if self._action is None:
            self._action = self.dist.sample().detach()
        return self.dist.log_prob(self._action)

    def entropy(self) -> Tensor:
        return self.dist.entropy()

    def set_action(self, action: Tensor) -> None:
        self._action = action


class CategoricalHead:
    """Categorical policy with no learnable parameter
    """
    def __call__(self, x: Tensor) -> Policy:
        return Policy(Categorical(logits=x))


class GaussinanHead(NetworkBlock):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        pass

    def forward(self, x: Tensor) -> Policy:
        return Policy(Categorical(logits=x))
