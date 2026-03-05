import torch.nn as nn

from .relu import ReLU
from .leakeyReLU import LeakyReLU


# Registry: activation_name -> class
activations = {
    "relu":       ReLU,
    "leaky_relu": LeakyReLU,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Instantiate an activation module by registry name.

    Args:
        name:   one of "relu", "leaky_relu"
        kwargs: forwarded to the activation class __init__
                (e.g. negative_slope=0.01 for LeakyReLU)

    Returns:
        nn.Module instance of the requested activation.
    """
    if name not in activations:
        raise ValueError(
            f"Unknown activation '{name}'. Available: {list(activations.keys())}"
        )
    return activations[name](**kwargs)
