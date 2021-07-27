"""
Utility for activation functions.
"""
from typing import Callable

import torch
import torch.nn.functional as F

import dynamics_toolbox.constants.activations as activations


def swish(x: torch.Tensor) -> torch.Tensor:
    """The swish activation function.
    Args:
         x: The input.
    Returns: Result of swish.
    """
    return x * torch.sigmoid(x)


def tanh(x: torch.Tensor) -> torch.Tensor:
    """The tanh activation function.
    Args:
         x: The input.
    Returns: Result of tanh.
    """
    return 0.5 * (torch.log(1 + x) - torch.log(1 - x))


ALL_ACTIVATIONS = {
    activations.RELU: F.relu,
    activations.SWISH: swish,
    activations.TANH: tanh,
}


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Get activation based on name.
    Args:
        name: The name of the activation.
    Returns: The activation.
    """
    return ALL_ACTIVATIONS[name]
