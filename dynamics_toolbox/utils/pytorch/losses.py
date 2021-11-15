"""
Utility for loss functions.
"""
from typing import Callable

import torch

import dynamics_toolbox.constants.losses as losses


def get_regression_loss(
        name: str,
        **kwargs
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Get a regression loss function based on name.

    Args:
        name: The name of the loss.
        kwargs: Any other named arguments to pass to the loss function.

    Returns:
        The loss function.
    """
    if name == losses.MSE:
        return torch.nn.MSELoss(**kwargs)
    elif name == losses.MAE:
        return torch.nn.L1Loss(**kwargs)
    else:
        raise ValueError(f'Unknown loss {name}.')
