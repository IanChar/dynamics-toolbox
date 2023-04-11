"""
A standard Q network.

Author: Ian Char
Date: April 6, 2023
"""
from typing import Callable, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from dynamics_toolbox.utils.pytorch.modules.fc_network import FCNetwork


class QNet(FCNetwork):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        hidden_activation: Callable[Tensor, Tensor] = F.relu,
        **kwargs
    ):
        """Constructor.

        obs_dim: Observation dimension size.
        act_dim: Action dimension size.
        hidden_sizes: Number of hidden units for each hidden layer.
        hidden_activation: Activation function.
        """
        super().__init__(
            input_dim=obs_dim + act_dim,
            output_dim=1,
            hidden_sizes=hidden_sizes,
            hidden_activation=hidden_activation,
        )

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        """Forward pass.

        Args:
            obs: Observation of shape (batch_size, obs_dim)
            act: Action of shape (batch_size, obs_dim).

        Returns: Value of shape (batch_size, 1)
        """
        return super().forward(torch.cat([obs, act], dim=-1))
