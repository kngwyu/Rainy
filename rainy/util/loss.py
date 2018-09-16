from torch import float32, Tensor

def mean_squared_loss(l: Tensor, r: Tensor) -> float32:
    return (l - r).pow(2).mul(0.5).mean()
