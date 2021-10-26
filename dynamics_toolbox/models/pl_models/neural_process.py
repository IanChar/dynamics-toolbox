"""
A model for the Neural Process as described in:
* https://arxiv.org/pdf/1807.01622.pdf
* https://arxiv.org/pdf/1807.01613.pdf
* https://arxiv.org/pdf/1901.05761.pdf

Author: Ian Char
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import numpy as np
import torch
from omegaconf import DictConfig, open_dict

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models import pl_models
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel


class NeuralProcess(AbstractPlModel):
    """A neural process model."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            condition_out_dim: int,
            latent_dim: int,
            conditioner_kwargs: DictConfig,
            latent_encoder_kwargs: DictConfig,
            decoder_kwargs: DictConfig,
            learning_rate: float = 1e-3,
            weight_decay: Optional[float] = 0.0,
            min_num_conditioning: int = 3,
            max_num_conditioning: int = 30,
            sample_mode: str = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
            **kwargs
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            condition_out_dim: The output dimension of the conditioning model.
            latent_dim: The dimensionality of the latent space.
            conditioner_kwargs: The config for the network that is responsible
                for conditioning. The output should be a deterministic value.
            latent_encoder_kwargs: The config for the latent encoding model. The output
                should be the mean and logvar of an independent multivariate Gaussian.
            decoder_kwargs: The config for the decoder model. The output is assumed
                to be deterministic (or more precisely, Gaussian with var=1).
            learning_rate: The learning rate for the network.
            weight_decay: The weight decay for the optimizer.
            min_num_conditioning: The minimum number of points to condition on
                when training.
            max_num_conditioning: The maximum number of points to condition on
                when training.
            sample_mode: The method to use for sampling.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self._min_num_conditioning = min_num_conditioning
        self._max_num_conditioning = max_num_conditioning
        self._sample_mode = sample_mode
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._condition_out_dim = condition_out_dim
        self._latent_dim = latent_dim
        self._curr_sample = None
        self._posterior_mean = torch.zeros(latent_dim)
        self._posterior_logvar = torch.zeros(latent_dim)
        with open_dict(conditioner_kwargs):
            conditioner_kwargs['input_dim'] = input_dim + output_dim
            conditioner_kwargs['output_dim'] = condition_out_dim
            conditioner_kwargs['learning_rate'] = learning_rate
        with open_dict(latent_encoder_kwargs):
            latent_encoder_kwargs['input_dim'] = condition_out_dim
            latent_encoder_kwargs['output_dim'] = latent_dim
            latent_encoder_kwargs['learning_rate'] = learning_rate
        with open_dict(decoder_kwargs):
            decoder_kwargs['input_dim'] = input_dim + latent_dim
            decoder_kwargs['output_dim'] = output_dim
            decoder_kwargs['learning_rate'] = learning_rate
        setattr(self, '_conditioner',
                getattr(pl_models,
                        conditioner_kwargs['model_type'])(**conditioner_kwargs))
        setattr(self, '_encoder',
                getattr(pl_models,
                        latent_encoder_kwargs['model_type'])(**latent_encoder_kwargs))
        setattr(self, '_decoder',
                getattr(pl_models,
                        decoder_kwargs['model_type'])(**decoder_kwargs))

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
        xi, yi = batch
        if len(xi.shape) == 2:
            xi = xi.unsqueeze(1)
            yi = yi.unsqueeze(1)
        pred_x, pred_y = xi[:, -1, :], yi[:, -1, :]
        if xi.shape[1] > 1:
            num_conditions = np.random.randint(
                min(self._min_num_conditioning, xi.shape[1] - 1),
                min(self._max_num_conditioning, xi.shape[1] - 1)
            )
            cond_idxs = torch.randperm(xi.shape[1] - 1)[:num_conditions]
            conditions = torch.cat([xi[:, cond_idxs, :], yi[:, cond_idxs, :]], dim=-1)
            condition_out = self._conditioner.forward(conditions).mean(dim=1)
            latent_out = self._encoder.get_net_out([condition_out, None])
            z_mu, z_logvar = latent_out['mean'], latent_out['logvar']
        else:
            condition_out = torch.zeros((xi.shape[0], self._condition_out_dim))
            z_mu, z_logvar = [torch.zeros((xi.shape[0], self._latent_dim))
                              for _ in range(2)]
        z_sample = torch.randn_like(z_mu) * (0.5 * z_logvar).exp() + z_mu
        decoder_in = torch.cat([pred_x, z_sample], dim=1)
        prediction = self._decoder.forward(decoder_in)
        return {
            'prediction': prediction,
            'condition_out': condition_out,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
        }

    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        _, yi = batch
        if len(yi.shape) > 2:
            yi = yi[:, -1, :]
        kldiv = torch.mean(
            -0.5 * torch.sum(1 + net_out['z_logvar'] - net_out['z_mu'].pow(2)
                             - net_out['z_logvar'].exp(), dim=1))
        # This is NLL with logvar=0
        mse = (net_out['prediction'] - yi).pow(2).mean()
        loss = kldiv + mse
        return loss, {'mse': mse.item(), 'kldiv': kldiv.item(), 'loss': loss.item()}

    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for next states and dictionary of info.
        """
        if (self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP
            or self._curr_sample is None):
            self._curr_sample = self._draw_from_posterior(1).to(self.device)
        latents = self._curr_sample[0].repeat(len(net_in)).reshape(len(net_in), -1)
        decoder_in = torch.cat([net_in, latents], dim=1)
        with torch.no_grad():
            predictions = self._decoder.forward(decoder_in)
        info = {'predictions': predictions, 'latents': latents}
        return predictions, info

    def multi_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
        torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            net_in: The input for the network.

        Returns:
            The predictions for next states and dictionary of info.
        """
        if (self._sample_mode == sampling_modes.SAMPLE_MEMBER_EVERY_STEP
            or self._curr_sample is None):
            self._curr_sample = self._draw_from_posterior(len(net_in)).to(self.device)
        elif len(self._curr_sample) < len(net_in):
            self._curr_sample = torch.cat(
                [self._curr_sample,
                 self._draw_from_posterior((len(net_in)
                     - len(self._curr_sample))).to(self.device)],
                dim=0)
        decoder_in = torch.cat([net_in, self._current_sample[:len(net_in)]], dim=1)
        with torch.no_grad():
            predictions = self._decoder.forward(decoder_in)
        info = {'predictions': predictions,
                'latents': self._current_sample[:len(net_in)]}
        return predictions, info

    def reset(self) -> None:
        """Reset the dynamics model."""
        self._curr_sample = None

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
            The mean and logvariance of the latent posterior.
        """
        condition_in = torch.cat([conditions_x, conditions_y], dim=1)
        with torch.no_grad():
            condition_out = self._conditioner.forward(condition_in)
            condition_out = condition_out.mean(dim=0).reshape(1, -1)
            latent_out = self._encoder.get_net_out([condition_out, None])
        self._posterior_mean = latent_out['mean']
        self._posterior_logvar = latent_out['logvar']
        return self._posterior_mean, self._posterior_logvar

    def clear_condition(self) -> None:
        """Clear the latent posterior and set back to the prior."""
        self._posterior_mean = torch.zeros(self._latent_dim)
        self._posterior_logvar = torch.zeros(self._latent_dim)

    def _draw_from_posterior(self, num_draws: int) -> torch.Tensor:
        """Draw samples from the latent posterior.

        Args:
            num_draws: The number of draws from the posterior to make.

        Returns:
            Tensor of the draws.
        """
        return (torch.randn((num_draws, self._latent_dim))
                * (0.5 * self._posterior_logvar).exp() + self._posterior_mean)

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
