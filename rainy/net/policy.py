from torch import Tensor
from torch.distributions import Categorical, Distribution


class Policy:
    def __init__(self, dist: Distribution):
        self.dist = dist
        self._action = None

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


def softmax(params: Tensor):
    return Policy(Categorical(logits=params))


