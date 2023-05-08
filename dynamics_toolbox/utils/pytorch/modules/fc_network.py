"""
A basic multi-layer perceptron implementation.

Author: Ian Char
Date: July, 11 2021
"""
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F


class FCNetwork(torch.nn.Module):
    """Fully Connected Network."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: Optional[Sequence[int]] = None,
        hidden_activation: Callable[[Tensor], Tensor] = F.relu,
        out_activation: Optional[Callable[[Tensor], Tensor]] = None,
        num_heads: int = 1,
    ):
        """Constructor.

        Args:
            input_dim: Dimension of input data.
            output_dim: Dimension of data outputted.
            hidden_sizes: List of sizes for the hidden layers.
            hidden_activation: Hidden activation function.
            out_activation: The activation function to apply on output. If there
                are multiple heads there should also be a corresponding amount of
                out activations.
        """
        super().__init__()
        assert (out_activation is None
                or num_heads == 1 or num_heads == len(out_activation))
        self._num_heads = num_heads
        if hidden_sizes is None:
            hidden_sizes = []
        if len(hidden_sizes) == 0:
            self._add_linear_layer(input_dim, output_dim, 0)
            self._n_layers = 1
        else:
            self._add_linear_layer(input_dim, hidden_sizes[0], 0)
            for hidx in range(len(hidden_sizes) - 1):
                self._add_linear_layer(hidden_sizes[hidx],
                                       hidden_sizes[hidx+1], hidx + 1)
            if self._num_heads == 1:
                self._add_linear_layer(hidden_sizes[-1], output_dim,
                                       len(hidden_sizes))
            else:
                for nh in range(self._num_heads):
                    self._add_linear_layer(hidden_sizes[-1], output_dim,
                                           layer_name=f'head_{nh}')
            self._n_layers = len(hidden_sizes) + 1
        self._hidden_activation = hidden_activation
        self._out_activation = out_activation

    def forward(
            self,
            net_in: Tensor,
    ) -> Tensor:
        """Forward pass through network.

        Args:
            net_in: The input to the network.

        Returns:
            The output of the network w shape (out_dim,) if one head or
            (n_heads, out_dim) if there are multiple heads.
        """
        curr = net_in
        for layer_num in range(self._n_layers - 1):
            curr = getattr(self, f'linear_{layer_num}')(curr)
            curr = self._hidden_activation(curr)
        if self._num_heads == 1:
            curr = getattr(self, f'linear_{self._n_layers - 1}')(curr)
            if self._out_activation is not None:
                return self._out_activation(curr)
            return curr
        currs = [getattr(self, f'head_{nh}')(curr) for nh in range(self._num_heads)]
        if self._out_activation is not None:
            currs = [outf(curr) for outf, curr in zip(self._out_activation, currs)]
        return torch.stack(currs)

    @property
    def n_layers(self) -> int:
        """Number of layers in the network."""
        return self._n_layers

    @property
    def hidden_activation(self) -> Callable[[Tensor], Tensor]:
        """Number of layers in the network."""
        return self._hidden_activation

    @property
    def out_activation(self) -> Callable[[Tensor], Tensor]:
        """Number of layers in the network."""
        return self._out_activation

    def get_layer(self, layer_num: int) -> torch.nn.Linear:
        """Return a specific layer."""
        return getattr(self, f'linear_{layer_num}')

    def _add_linear_layer(
            self,
            lin_in: int,
            lin_out: int,
            layer_num: Optional[int] = None,
            layer_name: Optional[str] = None,
    ) -> None:
        """Add a linear layer to the network.

        Args:
            lin_in: Input dimension to the layer.
            lin_out: Output dimension of the layer.
            layer_num: The number of the layer being added.
            layer_name: The name of the layer.
        """
        layer = torch.nn.Linear(lin_in, lin_out)
        if layer_name is None:
            if layer_num is None:
                raise ValueError('Either layer_num or layer_name must be provided')
            layer_name = f'linear_{layer_num}'
        self.add_module(layer_name, layer)
