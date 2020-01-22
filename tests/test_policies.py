from numpy.testing import assert_array_almost_equal as assert_array
import pytest
import torch
from typing import Callable

from rainy.net import policy as P


@pytest.mark.parametrize(
    "policy, check",
    [
        (P.BernoulliPolicy(logits=torch.randn(10)), lambda x: x == 0.0 or x == 1.0),
        (P.CategoricalPolicy(logits=torch.randn(10, 10)), lambda x: 0 <= x < 10,),
        (P.GaussianPolicy(torch.zeros(10), torch.ones(10)), lambda x: -10 <= x <= 10,),
        (
            P.TanhGaussianPolicy(torch.zeros(10), torch.ones(10)),
            lambda x: -1 <= x <= 1,
        ),
    ],
)
def test_action(policy: P.Policy, check: Callable[[float], bool]) -> None:
    action = policy.action()
    assert tuple(action.shape) == (10,)
    for x in action:
        assert check(x.item())


@pytest.mark.parametrize(
    "policy",
    [
        P.BernoulliPolicy(logits=torch.randn(10)),
        P.CategoricalPolicy(logits=torch.randn(10, 10)),
        P.GaussianPolicy(torch.zeros(10), torch.ones(10)),
        P.TanhGaussianPolicy(torch.zeros(10), torch.ones(10)),
    ],
)
def test_getitem(policy: P.Policy) -> None:
    best = policy.best_action()
    partial = policy[3:8].best_action()
    assert_array(best[3:8].numpy(), partial.numpy())


@pytest.mark.parametrize(
    "policy",
    [
        P.BernoulliPolicy(logits=torch.randn(10, 4)),
        P.CategoricalPolicy(logits=torch.randn(10, 10, 4)),
    ],
)
def test_merginalize(policy: P.Policy) -> None:
    sampled = policy.merginalize().sample()
    assert tuple(sampled.shape) == (10,)


@pytest.mark.parametrize(
    "policy",
    [
        P.GaussianPolicy(torch.zeros(10, requires_grad=True), torch.ones(10)),
        P.TanhGaussianPolicy(torch.zeros(10, requires_grad=True), torch.ones(10)),
    ],
)
def test_baction(policy: P.Policy) -> None:
    action = policy.baction()
    assert action.requires_grad


@pytest.mark.parametrize(
    "policy",
    [
        P.BernoulliPolicy(logits=torch.randn(10).requires_grad_()),
        P.CategoricalPolicy(logits=torch.randn(10, 10).requires_grad_()),
        P.GaussianPolicy(
            torch.zeros(10).requires_grad_(), torch.ones(10).requires_grad_()
        ),
        P.TanhGaussianPolicy(
            torch.zeros(10).requires_grad_(), torch.ones(10).requires_grad_()
        ),
    ],
)
def test_detach(policy: P.Policy) -> None:
    log_pi = policy.log_prob()
    assert log_pi.requires_grad
    detached = policy.detach()
    assert not detached.log_prob().requires_grad
