from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from torch.distributions import Bernoulli, Categorical, Distribution, Normal
from torch.nn import functional as F
from typing import Optional, Tuple
from ..prelude import Array, Index, Self
from ..utils import Device


class Policy(ABC):
    """Represents parameterized stochastic policies.
    """

    def __init__(self, dist: Distribution) -> None:
        self.dist = dist
        self._action: Optional[Tensor] = None
        self._baction: Optional[Tensor] = None

    def action(self) -> Tensor:
        """Sample actions if this policy has no action cache.
        """
        if self._action is None:
            self._action = self.sample()
        return self._action.squeeze()

    def baction(self) -> Tensor:
        """Sample "backwardable" actions.
        """
        if self._baction is None:
            self._baction = self.rsample()
        return self._baction.squeeze()

    def set_action(self, action: Tensor) -> None:
        self._action = action

    def sample(self) -> Tensor:
        return self.dist.sample().detach()

    def rsample(self) -> Tensor:
        """Sampling by reparameterization trick,
        which means the returned tensor is backwardable
        """
        return self.dist.rsample()

    def eval_action(self, deterministic: bool = False) -> Array:
        """Sample actions for evaluation, which leave no action cache
        and returns numpy array.
        """
        if deterministic:
            act = self.best_action()
        else:
            act = self.sample()
        return act.squeeze_().cpu().numpy()

    @abstractmethod
    def __getitem__(self, idx: Index) -> Self:
        pass

    @abstractmethod
    def best_action(self) -> Tensor:
        pass

    @abstractmethod
    def log_prob(self, use_baction: bool = False) -> Tensor:
        pass

    @abstractmethod
    def entropy(self) -> Tensor:
        pass


class BernoulliPolicy(Policy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Bernoulli(*args, **kwargs))
        self.dist: Bernoulli

    def best_action(self) -> Tensor:
        return self.dist.probs > 0.5

    def log_prob(self, use_baction: bool = False) -> Tensor:
        if use_baction:
            return self.dist.log_prob(self.baction())
        else:
            return self.dist.log_prob(self.action())

    def entropy(self) -> Tensor:
        return self.dist.entropy()

    def __getitem__(self, idx: Index) -> Self:
        return BernoulliPolicy(logits=self.dist.logits[idx])


class CategoricalPolicy(Policy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Categorical(*args, **kwargs))
        self.dist: Categorical

    def best_action(self) -> Tensor:
        return self.dist.probs.argmax(dim=-1)

    def log_prob(self, use_baction: bool = False) -> Tensor:
        if use_baction:
            return self.dist.log_prob(self.baction())
        else:
            return self.dist.log_prob(self.action())

    def entropy(self) -> Tensor:
        return self.dist.entropy()

    def __getitem__(self, idx: Index) -> Self:
        return CategoricalPolicy(logits=self.dist.logits[idx])


class GaussianPolicy(Policy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(Normal(*args, **kwargs))
        self.dist: Normal

    def best_action(self) -> Tensor:
        return self.dist.mean

    def entropy(self) -> Tensor:
        return self.dist.entropy().sum(-1)

    def log_prob(self, use_baction: bool = False) -> Tensor:
        if use_baction:
            return self.dist.log_prob(self.baction()).sum(-1)
        else:
            return self.dist.log_prob(self.action()).sum(-1)

    def __getitem__(self, idx: Index) -> Self:
        return GaussianPolicy(self.dist.mean[idx], self.dist.stddev[idx])


class TanhGaussianPolicy(GaussianPolicy):
    def __init__(self, *args, epsilon: float = 1e-6, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pre_tanh: Optional[Tensor] = None
        self.epsilon = epsilon

    def sample(self) -> Tensor:
        pre_tanh = self.dist.sample().detach()
        self._pre_tanh = pre_tanh
        return torch.tanh(pre_tanh)

    def rsample(self) -> Tensor:
        res = self.dist.rsample()
        self._pre_tanh = res
        return torch.tanh(res)

    def best_action(self) -> Tensor:
        return torch.tanh(self.dist.mean)

    def log_prob(self, use_baction: bool = False) -> Tensor:
        action = self._baction if use_baction else self._action
        if self._pre_tanh is None:
            pre_tanh = torch.log((1.0 + action) / (1.0 - action)) / 2
        else:
            pre_tanh = self._pre_tanh
        log_n = self.dist.log_prob(pre_tanh).sum(-1)
        log_da_du = torch.log(1.0 - action ** 2 + self.epsilon).sum(-1)
        return log_n - log_da_du


class PolicyDist(ABC, nn.Module):
    def __init__(self, action_dim: int, *args, **kwargs) -> None:
        super().__init__()
        self.action_dim = action_dim

    @property
    def input_dim(self) -> int:
        return self.action_dim

    @abstractmethod
    def forward(self, t: Tensor) -> Policy:
        pass


class BernoulliDist(PolicyDist):
    """Bernoulli policy with no learnable parameter
    """

    def forward(self, x: Tensor) -> Policy:
        return BernoulliPolicy(logits=x)


class CategoricalDist(PolicyDist):
    """Categorical policy with no learnable parameter
    """

    def forward(self, x: Tensor) -> Policy:
        return CategoricalPolicy(logits=x)


class GaussinanDist(PolicyDist):
    """Gaussian policy which takes both mean and stdev as inputs
    """

    @property
    def input_dim(self) -> int:
        return self.action_dim * 2

    def forward(self, x: Tensor) -> Policy:
        size = x.size(1) // 2
        mean, stddev = x[:, :size], x[:, size:]
        return GaussianPolicy(mean, F.softplus(stddev))


class TanhGaussianDist(GaussinanDist):
    """Tanh clipped Gaussian policy
    """

    def __init__(
        self,
        action_dim: int,
        logstd_range: Tuple[float, float] = (-20.0, 2.0),
        **kwargs,
    ) -> None:
        super().__init__(action_dim)
        self.logstd_range = logstd_range

    def forward(self, x: Tensor) -> Policy:
        size = x.size(1) // 2
        mu, logstd = x[:, :size], x[:, size:]
        std = torch.exp(logstd.clamp_(*self.logstd_range))
        return TanhGaussianPolicy(mu, std)


class SeparateStdGaussianDist(PolicyDist):
    """Gaussian policy which takes only mean as an input, and has a standard deviation
       independent with states, as a lernable parameter.
    """

    def __init__(self, action_dim: int, device: Device) -> None:
        super().__init__(action_dim)
        self.stddev = nn.Parameter(device.zeros(action_dim))

    def forward(self, x: Tensor) -> Policy:
        return GaussianPolicy(x, F.softplus(self.stddev))
