import torch

from .kaiming import kaiming_normal_, kaiming_uniform_
from .xavier import xavier_normal_, xavier_uniform_

# Registry: init_name -> callable(tensor) -> tensor
initializations = {
    "kaiming":         kaiming_normal_,
    "kaiming_normal":  kaiming_normal_,
    "kaiming_uniform": kaiming_uniform_,
    "xavier":          xavier_uniform_,
    "xavier_normal":   xavier_normal_,
    "xavier_uniform":  xavier_uniform_,
}


def uniform_(tensor: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """Fill tensor with values drawn from Uniform(a, b)."""
    with torch.no_grad():
        tensor.uniform_(a, b)
    return tensor


def constant_(tensor: torch.Tensor, val: float) -> torch.Tensor:
    """Fill tensor with a constant value."""
    with torch.no_grad():
        tensor.fill_(val)
    return tensor
