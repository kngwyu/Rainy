from abc import ABC, abstractmethod
from torch import nn, Tensor
from torch.distributions import Categorical, Distribution, Normal
from torch.nn import functional as F
from ..utils import Device


class Policy(ABC):
    def __init__(self, dist: Distribution) -> None:
        self.dist = dist
        self._action = None

    def detach_(self) -> None:
        self.dist.detach_()

    def action(self) -> Tensor:
        if self._action is None:
            self._action = self.dist.sample().detach()
        return self._action

    def set_action(self, action: Tensor) -> None:
        self._action = action

    @abstractmethod
    def best_action(self) -> Tensor:
        pass

    @abstractmethod
    def log_prob(self) -> Tensor:
        pass

    @abstractmethod
    def entropy(self) -> Tensor:
        pass


class CategoricalPolicy(Policy):
    def best_action(self) -> Tensor:
        return self.dist.probs.argmax(dim=-1)

    def log_prob(self) -> Tensor:
        return self.dist.log_prob(self.action())

    def entropy(self) -> Tensor:
        return self.dist.entropy()


class GaussianPolicy(Policy):
    def best_action(self) -> Tensor:
        return self.dist.mean

    def entropy(self) -> Tensor:
        return self.dist.entropy().sum(-1)

    def log_prob(self) -> Tensor:
        return self.dist.log_prob(self.action()).sum(-1)


class PolicyHead(ABC, nn.Module):
    def __init__(self, action_dim: int, *args, **kwargs) -> None:
        super().__init__()
        self.action_dim = action_dim

    @property
    def input_dim(self) -> int:
        return self.action_dim

    @abstractmethod
    def forward(self, t: Tensor) -> Policy:
        pass


class CategoricalHead(PolicyHead):
    """Categorical policy with no learnable parameter
    """
    def forward(self, x: Tensor) -> Policy:
        return CategoricalPolicy(Categorical(logits=x))


class GaussinanHead(PolicyHead):
    """Gaussian policy which takes both mean and stdev as inputs
    """
    @property
    def input_dim(self) -> int:
        return self.action_dim * 2

    def forward(self, x: Tensor) -> Policy:
        size = x.size(1) // 2
        mean, stddev = x[:, :size], x[:, size:]
        return GaussianPolicy(Normal(mean, F.softplus(stddev)))


class SeparateStdGaussinanHead(PolicyHead):
    """Gaussian policy which takes only mean as an input, and has a standard deviation
       independent with states, as a lernable parameter.
    """
    def __init__(self, action_dim: int, device: Device) -> None:
        super().__init__(action_dim)
        self.stddev = nn.Parameter(device.zeros(action_dim))

    def forward(self, x: Tensor) -> Policy:
        return GaussianPolicy(Normal(x, F.softplus(self.stddev)))
