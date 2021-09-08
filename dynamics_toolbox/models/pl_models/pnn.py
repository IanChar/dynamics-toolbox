"""
A network that outputs a Gaussian predictive distribution.

Author: Ian Char
"""
from typing import Optional, Tuple, Dict, Any, Sequence, Callable

import torch
import torch.nn.functional as F

import dynamics_toolbox.constants.activations as activations
from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
from dynamics_toolbox.utils.misc import s2i
from dynamics_toolbox.utils.pytorch.activations import get_activation
from dynamics_toolbox.utils.pytorch.fc_network import FCNetwork


class PNN(AbstractPlModel):
    """Two headed network that outputs mean and log variance of a Gaussian."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encoder_hidden_sizes: str,
            encoder_output_dim: int,
            mean_hidden_sizes: str,
            logvar_hidden_sizes: str,
            learning_rate: float = 1e-3,
            logvar_lower_bound: Optional[float] = None,
            logvar_upper_bound: Optional[float] = None,
            logvar_bound_loss_coef: float = 1e-3,
            hidden_activation: str = activations.SWISH,
            sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
            weight_decay: Optional[float] = 0.0,
            **kwargs,
    ):
        """
        Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            encoder_hidden_sizes: The sizes of the hidden layers of the encoder.
            encoder_output_dim: The output dimension of the encoder network.
            mean_hidden_sizes: Hidden layer sizes for mean head.
            logvar_hidden_sizes: Hidden layer sizes for logvar head.
            learning_rate: The learning rate for the network.
            logvar_lower_bound: Lower bound on the log variance. If none there is no bound.
            logvar_upper_bound: Lower bound on the log variance. If none there is no bound.
            logvar_bound_loss_coef: Coefficient on bound loss to add to loss.
            hidden_activation: Activation of the networks hidden layers.
            sample_mode: The method to use for sampling.
            weight_decay: The weight decay for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._encoder = FCNetwork(
            input_dim=input_dim,
            output_dim=encoder_output_dim,
            hidden_sizes=s2i(encoder_hidden_sizes),
            hidden_activation=get_activation(hidden_activation),
        )
        self._mean_head = FCNetwork(
            input_dim=encoder_output_dim,
            output_dim=output_dim,
            hidden_sizes=s2i(mean_hidden_sizes),
            hidden_activation=get_activation(hidden_activation)
        )
        self._logvar_head = FCNetwork(
            input_dim=encoder_output_dim,
            output_dim=output_dim,
            hidden_sizes=s2i(logvar_hidden_sizes),
            hidden_activation=get_activation(hidden_activation)
        )
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._var_pinning = logvar_lower_bound is not None and logvar_upper_bound is not None
        if self._var_pinning:
            self._min_logvar = torch.nn.Parameter(
                torch.Tensor([logvar_lower_bound])
                * torch.ones(1, output_dim, dtype=torch.float32, requires_grad=True))
            self._max_logvar = torch.nn.Parameter(
                torch.Tensor([logvar_upper_bound])
                * torch.ones(1, output_dim, dtype=torch.float32, requires_grad=True))
        else:
            self._min_logvar = None
            self._max_logvar = None
        self._logvar_bound_loss_coef = logvar_bound_loss_coef
        self._sample_mode = sample_mode

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function for network

        Args:
            x: The input to the network.

        Returns:
            The output of the networ.
        """
        encoded = self._encoder(x)
        mean, logvar = self._mean_head(encoded), self._logvar_head(encoded)
        if self._var_pinning:
            logvar = self._max_logvar - F.softplus(self._max_logvar - logvar)
            logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar

    def sample_model_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the next state as a delta in state.

        It is assumed that each input of the network should be drawn from a different
        sample.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        with torch.no_grad():
            mean_deltas, logvar_deltas = self.forward(net_in)
        std_deltas = (0.5 * logvar_deltas).exp()
        info = {'mean_deltas': mean_deltas, 'std_deltas': std_deltas}
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            deltas = torch.randn_like(mean_deltas) * std_deltas + mean_deltas
        else:
            deltas = mean_deltas
        return deltas, info

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """
        xi, _ = batch
        mean, logvar = self.forward(xi)
        return {'mean': mean, 'logvar': logvar}

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
        _, labels = batch
        mean = net_out['mean']
        logvar = net_out['logvar']
        sq_diffs = (mean - labels) ** 2
        mse = torch.mean(sq_diffs)
        loss = torch.mean(torch.exp(-logvar) * sq_diffs + logvar)
        stats = dict(
            nll=loss.item(),
            mse=mse.item(),
            dynamics_mse=torch.mean((mean[:, 1:] - labels[:, 1:]) ** 2).item(),
            rewards_mse=torch.mean((mean[:, 0] - labels[:, 0]) ** 2).item(),
        )
        stats['logvar/mean'] = logvar.mean().item()
        if self._var_pinning:
            bound_loss = self._logvar_bound_loss_coef *\
                         torch.abs(self._max_logvar - self._min_logvar).mean()
            stats['bound_loss'] = bound_loss.item()
            stats['logvar_lower_bound/mean'] = self._min_logvar.mean().item()
            stats['logvar_upper_bound/mean'] = self._max_logvar.mean().item()
            stats['logvar_bound_difference'] = (self._max_logvar - self._min_logvar).mean().item()
            loss += bound_loss
        stats['loss'] = loss.item()
        return loss, stats

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer"""
        return torch.optim.Adam(self.parameters(), lr=self._learning_rate)

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        if self._sample_mode not in [sampling_modes.SAMPLE_FROM_DIST, sampling_modes.RETURN_MEAN]:
            raise ValueError(f'PNN sample mode must either be {sampling_modes.SAMPLE_FROM_DIST} '
                             f'or {sampling_modes.RETURN_MEAN}, but received {mode}.')
        self._sample_mode = mode

    @property
    def input_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._input_dim

    @property
    def output_dim(self) -> int:
        """The sample mode is the method that in which we get next state."""
        return self._output_dim

    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        """Get the list of metric functions to compute."""
        return {}

    @property
    def learning_rate(self) -> float:
        """Get the learning rate."""
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        """Get the weight decay."""
        return self._weight_decay
