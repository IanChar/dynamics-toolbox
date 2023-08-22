"""
Utility for activation functions.
"""
from typing import Callable

import torch
import torch.nn.functional as F

import dynamics_toolbox.constants.activations as activations


def swish(x: torch.Tensor) -> torch.Tensor:
    """The swish activation function.

    swish(x) = x * sigmoid(x)

    Args:
         x: The input.

    Returns:
        Result of swish.
    """
    return x * torch.sigmoid(x)


def identity(x: torch.Tensor) -> torch.Tensor:
    """Just returns the same thing."""
    return x


ALL_ACTIVATIONS = {
    activations.RELU: F.relu,
    activations.SWISH: swish,
    activations.TANH: torch.tanh,
    activations.IDENTITY: identity,
    activations.SIGMOID: F.sigmoid,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get activation based on name.

    Args:
        name: The name of the activation.

    Returns:
        The activation.
    """
    if name is None:
        return None
    return ALL_ACTIVATIONS[name]
