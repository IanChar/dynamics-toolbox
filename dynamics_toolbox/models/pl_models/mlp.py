"""
Standard multi-layer perceptron dynamics model.

Author: Ian Char
"""
from typing import Sequence, NoReturn, Tuple, Dict, Any

import torch

from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
import dynamics_toolbox.constants.activations as activations
from dynamics_toolbox.utils.misc import s2i
from dynamics_toolbox.utils.pytorch.activations import get_activation
from dynamics_toolbox.utils.pytorch.torch_mlp import TorchMlp


class MLP(AbstractPlModel):
    """Fully connected network for dynamics."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_sizes: str,
            learning_rate: float,
            hidden_activation: str = activations.RELU,
            **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            learning_rate: The learning rate for the network.
            hidden_sizes: String of comma separated ints for the hidden sizes. e.g. 256,256,256
        """
        super().__init__()
        self.save_hyperparameters()
        self._net = TorchMlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=s2i(hidden_sizes),
            hidden_activation=get_activation(hidden_activation),
        )
        self._learning_rate = learning_rate
        self._sample_mode = ''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for network

        Args:
            x: The input to the network.

        Returns:
            The output of the networ.
        """
        return self._net.forward(x)

    def _get_deltas_from_torch(self, net_in: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the delta in state

        Args:
            net_in: The input for the network.

        Returns:
            The next states and dictionary of info.
        """
        with torch.no_grad():
            deltas = self.forward(net_in)
        info = {'delta': deltas}
        return deltas, info

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """
        xi, _ = batch
        output = self.forward(xi)
        return {'prediction': output}

    def loss(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            The loss and a dictionary of other statistics.
        """
        _, yi = batch
        l2 = ((net_out['prediction'] - yi) ** 2).sum(dim=1).mean()
        stats = {'loss': l2.item()}
        return l2, stats

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> NoReturn:
        """Set the sample mode to the appropriate mode."""
        self._sample_mode = mode

    @property
    def input_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._output_dim
