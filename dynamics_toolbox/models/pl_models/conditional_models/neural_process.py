"""
A model for the Neural Process as described in:
* https://arxiv.org/pdf/1807.01622.pdf
* https://arxiv.org/pdf/1807.01613.pdf
* https://arxiv.org/pdf/1901.05761.pdf

Author: Ian Char
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


class NeuralProcess(AbstractConditionalModel):
    """A neural process model."""

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            condition_out_dim: int,
            latent_dim: int,
            conditioner_cfg: DictConfig,
            latent_encoder_cfg: DictConfig,
            decoder_cfg: DictConfig,
            condition_sampler_cfg: DictConfig,
            beta: float = 1,
            learning_rate: float = 1e-3,
            weight_decay: Optional[float] = 0.0,
            sample_mode: str = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
            **kwargs
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            condition_out_dim: The output dimension of the conditioning model.
            latent_dim: The dimensionality of the latent space.
            conditioner_cfg: The config for the network that is responsible
                for conditioning. This should be a DatasetEncoder.
            latent_encoder_cfg: The config for the latent encoding model. The output
                should be the mean and logvar of an independent multivariate Gaussian.
            decoder_cfg: The config for the decoder model. The output is assumed
                to be deterministic (or more precisely, Gaussian with var=1).
            condition_sampler_cfg: The config for the condition sampling. Must be
                a config for a ConditionSampler.
            beta: The coefficient to weight the KL divergence by.
            learning_rate: The learning rate for the network.
            weight_decay: The weight decay for the optimizer.
            sample_mode: The method to use for sampling.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self._sample_mode = sample_mode
        self._beta = beta
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._condition_out_dim = condition_out_dim
        self._latent_dim = latent_dim
        self._curr_sample = None
        self._posterior_mean = torch.zeros(latent_dim)
        self._posterior_logvar = torch.zeros(latent_dim)
        self._conditioner = hydra.utils.instantiate(
            conditioner_cfg,
            input_dim=input_dim + output_dim,
            output_dim=condition_out_dim,
            _recursive_=False,
        )
        assert isinstance(self._conditioner, DatasetEncoder), \
            'Conditioner must be a DatasetEncoder.'
        self._encoder = hydra.utils.instantiate(
            latent_encoder_cfg,
            input_dim=condition_out_dim,
            output_dim=latent_dim,
            _recursive_=False,
        )
        self._decoder = hydra.utils.instantiate(
            decoder_cfg,
            input_dim=input_dim + latent_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        self._condition_sampler = hydra.utils.instantiate(
            condition_sampler_cfg,
            _recursive_=False,
        )
        assert isinstance(self._condition_sampler, ConditionSampler), \
            'Condition sampler must be a ConditionSampler.j'

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
            latent_out = self._encoder.get_net_out([condition_out, None])
            z_mu, z_logvar = latent_out['mean'], latent_out['logvar']
        else:
            condition_out = torch.zeros((batch[0].shape[0], self._condition_out_dim))
            z_mu, z_logvar = [torch.zeros((batch[0].shape[0], self._latent_dim))
                              for _ in range(2)]
        z_sample = torch.randn_like(z_mu).to(self.device) * (0.5 * z_logvar).exp() \
                   + z_mu
        decoder_in = torch.cat([pred_x, z_sample], dim=1)
        prediction = self._decoder.forward(decoder_in)
        return {
            'prediction': prediction,
            'condition_out': condition_out,
            'z_mu': z_mu,
            'z_logvar': z_logvar,
            'label': pred_y,
        }

    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        yi = net_out['label']
        if len(yi.shape) > 2:
            yi = yi[:, -1, :]
        kldiv = torch.mean(
            -0.5 * torch.sum(1 + net_out['z_logvar'] - net_out['z_mu'].pow(2)
                             - net_out['z_logvar'].exp(), dim=1))
        # This is NLL with logvar=0
        mse = (net_out['prediction'] - yi).pow(2).mean()
        loss = self._beta * kldiv + mse
        return loss, {
            'mse': mse.item(),
            'kldiv': kldiv.item(),
            'loss': loss.item(),
            'latent_std_magnitude': (0.5 * net_out['z_logvar']).exp().mean().item(),
        }

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
        decoder_in = torch.cat([net_in, self._curr_sample[:len(net_in)]], dim=1)
        with torch.no_grad():
            predictions = self._decoder.forward(decoder_in)
        info = {'predictions': predictions,
                'latents': self._curr_sample[:len(net_in)]}
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
        condition_in = torch.cat([conditions_x, conditions_y], dim=1).unsqueeze(0)
        with torch.no_grad():
            condition_out = self._conditioner.encode_dataset(condition_in)
            latent_out = self._encoder.get_net_out([condition_out, None])
        self._posterior_mean = latent_out['mean']
        self._posterior_logvar = latent_out['logvar']
        return self._posterior_mean, self._posterior_logvar

    def clear_condition(self) -> None:
        """Clear the latent posterior and set back to the prior."""
        self._posterior_mean = torch.zeros(self._latent_dim).to(self.device)
        self._posterior_logvar = torch.zeros(self._latent_dim).to(self.device)

    def _draw_from_posterior(self, num_draws: int) -> torch.Tensor:
        """Draw samples from the latent posterior.

        Args:
            num_draws: The number of draws from the posterior to make.

        Returns:
            Tensor of the draws.
        """
        return (torch.randn((num_draws, self._latent_dim)).to(self.device)
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
