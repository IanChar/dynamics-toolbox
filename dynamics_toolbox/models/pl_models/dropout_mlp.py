"""
MLP but dropout is sampled to get different networks.

Model based on https://arxiv.org/abs/1506.02142

Author: Ian Char
"""
from typing import Sequence, Tuple, Dict, Any, Optional, List, Callable

import torch
from torchmetrics import ExplainedVariance

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
import dynamics_toolbox.constants.activations as activations
import dynamics_toolbox.constants.losses as losses
from dynamics_toolbox.utils.misc import s2i
from dynamics_toolbox.utils.pytorch.activations import get_activation
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss
from dynamics_toolbox.utils.pytorch.fc_network import FCNetwork


class DropoutMLP(AbstractPlModel):
    """Fully connected network where simplex of weights are low loss."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            learning_rate: float,
            dropout_prob: float,
            num_layers: Optional[int] = None,
            layer_size: Optional[int] = None,
            architecture: Optional[str] = None,
            hidden_activation: str = activations.RELU,
            loss_type: str = losses.MSE,
            sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
            weight_decay: Optional[float] = 0.0,
            **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            learning_rate: The learning rate for the network.
            dropout_prob: The probability of dropping out a connection.
            num_layers: The number of hidden layers in the MLP.
            layer_size: The size of each hidden layer in the MLP.
            architecture: The architecture of the MLP described as a
                a string of underscore separated ints e.g. 256_100_64.
                If provided, this overrides num_layers and layer_sizes.
            hidden_activation: Activation to use.
            loss_type: The name of the loss function to use.
            sample_mode: The type of sampling to perform.
            weight_decay: The weight decay for the optimizer.
        """

        super().__init__()
        self.save_hyperparameters()
        if architecture is not None:
            hidden_sizes = s2i(architecture)
        elif num_layers is not None and layer_size is not None:
            hidden_sizes = [layer_size for _ in range(num_layers)]
        else:
            raise ValueError(
                'MLP architecture not provided. Either specify architecture '
                'argument or both num_layers and layer_size arguments.'
            )
        self._net = FCNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_activation=get_activation(hidden_activation),
        )
        # Instantiate dropout layers.
        for layer in range(1, self._net.n_layers):
            setattr(self, f'_dropout_{layer}', torch.nn.Dropout(dropout_prob))
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._dropout_prob = dropout_prob
        self._loss_function = get_regression_loss(loss_type)
        self._loss_type = loss_type
        self._sample_mode = sample_mode
        self._curr_sample = None
        # Need this to keep the mask so that it is saved when the backprop step is
        # done outside of the code.
        self._dropout_dist = torch.distributions.bernoulli.Bernoulli(1 - dropout_prob)
        # TODO: In the future we may want to pass this in as an argument.
        self._metrics = {
            'EV': ExplainedVariance(),
            'IndvEV': ExplainedVariance('raw_values'),
        }

    def reset(self) -> None:
        """Reset the dynamics model."""
        self._curr_sample = None

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function for network.

        Args:
            x: The input to the network.

        Returns:
            The output of the network.
        """
        curr = self._net.linear_0(x)
        for layer in range(1, self._net.n_layers - 1):
            curr = getattr(self, f'_dropout_{layer}')(curr)
            curr = getattr(self._net, f'linear_{layer}')(curr)
            curr = self._net.hidden_activation(curr)
        curr = getattr(self, f'_dropout_{self._net.n_layers - 1}')(curr)
        curr = getattr(self._net, f'linear_{self._net.n_layers - 1}')(curr)
        if self._net.out_activation is not None:
            return self._net.out_activation(curr)
        return curr

    def single_sample_output_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        if (self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP
                or self._curr_sample is None):
            self._curr_sample = self._sample_dropout_mask(len(net_in))
        masks = [cs[0].repeat(len(net_in)).reshape(len(net_in), -1)
                 for cs in self._curr_sample]
        deltas = self._forward_with_specified_mask(net_in, masks)
        info = {'delta': deltas}
        return deltas, info

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
        if (self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP
                or self._curr_sample is None):
            self._curr_sample = self._sample_dropout_mask(len(net_in))
        elif len(self._curr_sample) < len(net_in):
            additional_masks = self._sample_dropout_mask(len(net_in)
                                                         - len(self._curr_sample))
            self._curr_sample = [torch.cat([csamp, asamp], dim=0)
                                 for csamp, asamp in zip(self._curr_sample,
                                                         additional_masks)]
        masks = [cs[:len(net_in)] for cs in self._curr_sample]
        deltas = self._forward_with_specified_mask(net_in, masks)
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

    def _sample_dropout_mask(
            self,
            num_inputs: int,
    ) -> List[torch.Tensor]:
        """Sample a dropout mask.

        Args:
            num_inputs: The number of inputs we should draw samples for.

        Returns:
            List of tensors of 0s and 1s with shape (num_inputs, input_layer_size).
            Starting with the second layer.
        """
        masks = []
        for lidx in range(1, self._net.n_layers):
            masks.append(self._dropout_dist.sample((
                num_inputs,
                getattr(self._net, f'linear_{lidx}').in_features)).to(self.device))
        return masks

    def _forward_with_specified_mask(
            self,
            x: torch.Tensor,
            masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """Forward function for network with a prespecified list of masks.

        This operation does not keep track of gradients.

        Args:
            x: The input to the network.
            masks: The mask for connections for each layer and each input. The length of
                the list must be the same as the number of layers in the network. Each
                member of the list has shape (x.shape[0], num_inputs_at_layer).

        Returns:
            The output of the network.
        """
        with torch.no_grad():
            curr = self._net.linear_0(x)
            for layer, mask in enumerate(masks[:-1]):
                curr *= mask
                curr = getattr(self._net, f'linear_{layer + 1}')(curr)
                curr = self._net.hidden_activation(curr)
            curr *= masks[-1]
            curr = getattr(self._net, f'linear_{self._net.n_layers - 1}')(curr)
        if self._net.out_activation is not None:
            return self._net.out_activation(curr)
        return curr
