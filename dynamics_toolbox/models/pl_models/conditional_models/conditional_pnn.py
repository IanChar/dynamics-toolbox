"""
Conditional network that outputs the mean and gaussian.

Author: Ian Char
Date: 11/7/2021
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import hydra.utils
import torch
from omegaconf import DictConfig

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.conditional_models.abstract_conditional_model import \
    AbstractConditionalModel
from dynamics_toolbox.utils.pytorch.condition_sampler import ConditionSampler
from dynamics_toolbox.utils.pytorch.modules.dataset_encoder import DatasetEncoder


class ConditionalPNN(AbstractConditionalModel):
    """A neural process model."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            latent_mean_dim: int,
            latent_logvar_dim: int,
            conditioner_cfg: DictConfig,
            mean_net_cfg: DictConfig,
            logvar_net_cfg: DictConfig,
            condition_sampler_cfg: DictConfig,
            learning_rate: float = 1e-3,
            logvar_lower_bound: Optional[float] = None,
            logvar_upper_bound: Optional[float] = None,
            logvar_bound_loss_coef: float = 1e-3,
            weight_decay: Optional[float] = 0.0,
            sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
            **kwargs
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            latent_mean_dim: The latent parameter produced for the mean.
            latent_logvar_dim: The latent parameter produced for the log variance.
            conditioner_cfg: The config for the network that is responsible
                for conditioning. This should be a DatasetEncoder.
            mean_net_cfg: The configuration for the network outputting a mean.
            logvar_net_cfg: The configuration for the network outputting the
                logvariance.
            learning_rate: The learning rate for the network.
            logvar_lower_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_upper_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_bound_loss_coef: Coefficient on bound loss to add to loss.
            weight_decay: The weight decay for the optimizer.
            sample_mode: The method to use for sampling.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self._sample_mode = sample_mode
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._latent_mean_dim = latent_mean_dim
        self._latent_logvar_dim = latent_logvar_dim
        self._mean_condition = torch.zeros(latent_mean_dim).to(self.device)
        self._logvar_condition = torch.zeros(latent_logvar_dim).to(self.device)
        self._conditioner = hydra.utils.instantiate(
            conditioner_cfg,
            input_dim=input_dim + output_dim,
            output_dim=latent_mean_dim + latent_logvar_dim,
            _recursive_=False,
        )
        assert isinstance(self._conditioner, DatasetEncoder), \
            'Conditioner must be a DatasetEncoder.'
        self._mean_net = hydra.utils.instantiate(
            mean_net_cfg,
            input_dim=input_dim + latent_mean_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        self._logvar_net = hydra.utils.instantiate(
            logvar_net_cfg,
            input_dim=input_dim + latent_logvar_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        self._condition_sampler = hydra.utils.instantiate(
            condition_sampler_cfg,
            _recursive_=False,
        )
        assert isinstance(self._condition_sampler, ConditionSampler), \
            'Condition sampler must be a ConditionSampler.j'
        self._var_pinning = logvar_lower_bound is not None \
                            and logvar_upper_bound is not None
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

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network. Since this is a meta learning
                algorithm. It is expected that batch has 2 tensors each with shape
                (num_datasets, num_points, in/out dim). If the shape is 2, it is
                assumed that each point is from a different dataset. The last point
                in the sequence i.e. (:, -1, :) is assume to be the point we are
                predicting for.

        Returns:
            Dictionary of name to tensor.
        """
        conditions, pred_x, pred_y = self._condition_sampler.split_batch(batch)
        if conditions is not None:
            condition_out = self._conditioner.encode_dataset(conditions)
        else:
            condition_out = torch.zeros((batch[0].shape[0], self._condition_out_dim))
        mean_condition = condition_out[..., :self._latent_mean_dim]
        logvar_condition = condition_out[..., self._latent_mean_dim:]
        mean_prediction = self._mean_net(torch.cat([pred_x, mean_condition], dim=1))
        logvar_prediction = self._logvar_net(
            torch.cat([pred_x, logvar_condition], dim=1))
        return {
            'mean': mean_prediction,
            'logvar': logvar_prediction,
            'label': pred_y,
        }

    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        labels = net_out['label']
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

    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for next states and dictionary of info.
        """
        with torch.no_grad():
            mean_predictions = self._mean_net(torch.cat([
                net_in,
                self._mean_condition.reshape(1, -1).repeat(len(net_in), 1)], dim=1)
            )
            logvar_predictions = self._logvar_net(torch.cat([
                net_in,
                self._logvar_condition.reshape(1, -1).repeat(len(net_in), 1)], dim=1)
            )
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

    def multi_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for next states and dictionary of info.
        """
        return self.single_sample_output_from_torch(net_in)

    def condition_samples(
            self,
            conditions_x: torch.Tensor,
            conditions_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Set the latent posterior of the neural process based on data observed.

        Args:
            conditions_x: The x points to condition on.
            conditions_y: The corresponding y points to condition on.

        Returns:
            The mean and logvariance of the latent encodings.
        """
        condition_in = torch.cat([conditions_x, conditions_y], dim=1).unsqueeze(0)
        with torch.no_grad():
            condition_out = self._conditioner.encode_dataset(condition_in)
        self._mean_condition = condition_out[..., :self._latent_mean_dim]
        self._logvar_condition = condition_out[..., self._latent_mean_dim:]
        return self._mean_condition, self._logvar_condition

    def clear_condition(self) -> None:
        """Clear the latent posterior and set back to the prior."""
        self._mean_condition = torch.zeros(self._latent_mean_dim).to(self.device)
        self._logvar_condition = torch.zeros(self._latent_logvar_dim).to(self.device)

    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        return {}

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @property
    def sample_mode(self) -> str:
        return self._sample_mode

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

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
