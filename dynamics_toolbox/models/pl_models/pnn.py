"""
A network that outputs a Gaussian predictive distribution.

Author: Ian Char
"""
from typing import Optional, Tuple, Dict, Any, Sequence, Callable

import hydra.utils
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import ExplainedVariance

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel


class PNN(AbstractPlModel):
    """Two headed network that outputs mean and log variance of a Gaussian."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encoder_output_dim: int,
            encoder_cfg: DictConfig,
            mean_net_cfg: DictConfig,
            logvar_net_cfg: DictConfig,
            learning_rate: float = 1e-3,
            logvar_lower_bound: Optional[float] = None,
            logvar_upper_bound: Optional[float] = None,
            logvar_bound_loss_coef: float = 1e-3,
            sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
            weight_decay: Optional[float] = 0.0,
            **kwargs,
    ):
        """
        Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            encoder_output_dim: The dimension of the encoder to output.
            encoder_cfg: Configuration for the encoder. The object created must have
                a forward method.
            mean_net_cfg: Configuration for the mean. The object created must have
                a forward method.
            logvar_net_cfg: Configuration for the logvar. The object created must have
                a forward method.
            learning_rate: The learning rate for the network.
            logvar_lower_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_upper_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_bound_loss_coef: Coefficient on bound loss to add to loss.
            hidden_activation: Activation of the networks hidden layers.
            sample_mode: The method to use for sampling.
            weight_decay: The weight decay for the optimizer.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.save_hyperparameters()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._encoder = hydra.utils.instantiate(
            encoder_cfg,
            input_dim=input_dim,
            output_dim=encoder_output_dim,
            _recursive_=False,
        )
        self._mean_head = hydra.utils.instantiate(
            mean_net_cfg,
            input_dim=encoder_output_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        self._logvar_head = hydra.utils.instantiate(
            logvar_net_cfg,
            input_dim=encoder_output_dim,
            output_dim=output_dim,
            _recursive_=False,
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
        self._metrics = {
                'EV': ExplainedVariance(),
                'IndvEV': ExplainedVariance('raw_values'),
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function for network

        Args:
            x: The input to the network.

        Returns:
            The output of the networ.
        """
        encoded = self._encoder.forward(x)
        mean = self._mean_head.forward(encoded)
        logvar = self._logvar_head.forward(encoded)
        if self._var_pinning:
            logvar = self._max_logvar - F.softplus(self._max_logvar - logvar)
            logvar = self._min_logvar + F.softplus(logvar - self._min_logvar)
        return mean, logvar

    def single_sample_output_from_torch(
            self,
            net_in: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for next states and dictionary of info.
        """
        with torch.no_grad():
            mean_predictions, logvar_predictions = self.forward(net_in)
        std_predictions = (0.5 * logvar_predictions).exp()
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            predictions = (torch.randn_like(mean_predictions) * std_predictions
                           + mean_predictions)
        else:
            predictions = mean_predictions
        info = {'predictions': predictions,
                'mean_predictions': mean_predictions,
                'std_predictions': std_predictions}
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
        sq_diffs = (mean - labels).pow(2)
        mse = torch.mean(sq_diffs)
        loss = torch.mean(torch.exp(-logvar) * sq_diffs + logvar)
        stats = dict(
            nll=loss.item(),
            mse=mse.item(),
        )
        stats['logvar/mean'] = logvar.mean().item()
        if self._var_pinning:
            bound_loss = self._logvar_bound_loss_coef * \
                         torch.abs(self._max_logvar - self._min_logvar).mean()
            stats['bound_loss'] = bound_loss.item()
            stats['logvar_lower_bound/mean'] = self._min_logvar.mean().item()
            stats['logvar_upper_bound/mean'] = self._max_logvar.mean().item()
            stats['logvar_bound_difference'] = (
                        self._max_logvar - self._min_logvar).mean().item()
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
        if self._sample_mode not in [sampling_modes.SAMPLE_FROM_DIST,
                                     sampling_modes.RETURN_MEAN]:
            raise ValueError(
                f'PNN sample mode must either be {sampling_modes.SAMPLE_FROM_DIST} '
                f'or {sampling_modes.RETURN_MEAN}, but received {mode}.')
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
        return super()._get_test_and_validation_metrics(
            {'prediction': net_out['mean']},
            batch,
        )
