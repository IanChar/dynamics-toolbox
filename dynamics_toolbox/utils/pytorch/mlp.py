"""
A basic multi-layer perceptron implementation.

Author: Ian Char
Date: July, 11 2021
"""
from typing import Callable, NoReturn, Optional, Sequence

import torch
import torch.nn.functional as F


class MLP(torch.nn.Module):
    """MLP Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Sequence[int],
        hidden_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        out_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        """Constructor.
        Args:
            input_dim: Dimension of input data.
            output_dim: Dimension of data outputted.
            hidden_sizes: List of sizes for the hidden layers.
            hidden_activation: Hidden activation function.
            out_activation: The activation function to apply on output.
        """
        super().__init__()
        if len(hidden_sizes) == 0:
            self._add_linear_layer(input_dim, output_dim, 0)
            self.n_layers = 1
        else:
            self._add_linear_layer(input_dim, hidden_sizes[0], 0)
            for hidx in range(len(hidden_sizes) - 1):
                self._add_linear_layer(hidden_sizes[hidx],
                                       hidden_sizes[hidx+1], hidx + 1)
            self._add_linear_layer(hidden_sizes[-1], output_dim,
                                   len(hidden_sizes))
            self.n_layers = len(hidden_sizes) + 1
        self.hidden_activation = hidden_activation
        self.out_activation = out_activation

    def forward(
            self,
            net_in: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through network.
        Args:
            net_in: The input to the network.
        Returns: The output of the network."""
        curr = net_in
        for layer_num in range(self.n_layers - 1):
            curr = getattr(self, 'linear_%d' % layer_num)(curr)
            curr = self.hidden_activation(curr)
        curr = getattr(self, 'linear_%d' % (self.n_layers - 1))(curr)
        if self.out_activation is not None:
            return self.out_activation(curr)
        return curr

    def _add_linear_layer(
            self,
            lin_in: int,
            lin_out: int,
            layer_num: int,
    ) -> NoReturn:
        """Add a linear layer to the network.
        Args:
            lin_in: Input dimension to the layer.
            lin_out: Output dimension of the layer.
            layer_num: The number of the layer being added.
        """
        layer = torch.nn.Linear(lin_in, lin_out)
        self.add_module('linear_%d' % layer_num, layer)
