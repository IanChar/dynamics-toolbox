"""
Standard multi-layer perceptron dynamics model.

Author: Ian Char
"""
from typing import Sequence, Tuple, Dict, Any, Optional, Callable

import torch
from torchmetrics import ExplainedVariance

from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
import dynamics_toolbox.constants.activations as activations
import dynamics_toolbox.constants.losses as losses
from dynamics_toolbox.utils.misc import get_architecture
from dynamics_toolbox.utils.pytorch.activations import get_activation
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss
from dynamics_toolbox.utils.pytorch.modules.fc_network import FCNetwork


class MLP(AbstractPlModel):
    """Fully connected network for dynamics."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            embed_dim: int,
            num_layers_per_block: int,
            num_blocks: int,
            learning_rate: float = 3e-4,
            hidden_activation: str = activations.RELU,
            loss_type: str = losses.MSE,
            weight_decay: Optional[float] = 1e-3,
            **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            learning_rate: The learning rate for the network.
            num_layers: The number of hidden layers in the MLP.
            layer_size: The size of each hidden layer in the MLP.
            architecture: The architecture of the MLP described as a
                a string of underscore separated ints e.g. 256_100_64.
                If provided, this overrides num_layers and layer_sizes.
            hidden_activation: Activation to use.
            loss_type: The name of the loss function to use.
            weight_decay: The weight decay for the optimizer.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.num_blocks = num_blocks
        hidden_sizes = get_architecture(num_layers_per_block, embed_dim, None)
        for bnum in range(num_blocks):
            outdim = embed_dim if bnum < num_blocks - 1 else output_dim
            setattr(self, f'block_{bnum}', FCNetwork(
                input_dim=input_dim,
                output_dim=outdim,
                hidden_sizes=hidden_sizes,
                hidden_activation=get_activation(hidden_activation),
            ))
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = ''
        self._loss_function = get_regression_loss(loss_type)
        self._loss_type = loss_type
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
                'EV': ExplainedVariance(),
                'IndvEV': ExplainedVariance('raw_values'),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for network

        Args:
            x: The input to the network.

        Returns:
            The output of the networ.
        """
        for bnum in range(self.num_blocks):
            x = x + getattr(self, f'block_{bnum}')(x)
        return x

    def single_sample_output_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for a single function sample
        """
        with torch.no_grad():
            predictions = self.forward(net_in)
        info = {'predictions': predictions}
        return predictions, info

    def multi_sample_output_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        return self.single_sample_output_from_torch(net_in)

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
        loss = self._loss_function(net_out['prediction'], yi)
        stats = {'loss': loss.item()}
        return loss, stats

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        self._sample_mode = mode

    @property
    def input_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._hparams.input_dim

    @property
    def output_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._hparams.output_dim

    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        """Get the list of metric functions to compute."""
        return self._metrics

    @property
    def learning_rate(self) -> float:
        """Get the learning rate."""
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        """Get the weight decay."""
        return self._weight_decay

    def _get_test_and_validation_metrics(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute additional metrics to be used for validation/test only.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            A dictionary of additional metrics.
        """
        to_return = {}
        pred = net_out['prediction']
        _, yi = batch
        for metric_name, metric in self._metrics.items():
            metric_value = metric(pred, yi)
            if len(metric_value.shape) > 0:
                for dim_idx, metric_v in enumerate(metric_value):
                    to_return[f'{metric_name}_dim{dim_idx}'] = metric_v
            else:
                to_return[metric_name] = metric_value
        return to_return
