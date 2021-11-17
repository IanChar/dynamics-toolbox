"""
A network that takes in the input vector and quantile levels and outputs
"""

import os, sys
from typing import Optional, Sequence, Callable, Tuple, Any, Dict

import hydra.utils
from argparse import Namespace
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig

import dynamics_toolbox.constants.activations as activations
from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
from dynamics_toolbox.utils.misc import get_architecture
from dynamics_toolbox.utils.pytorch.losses import get_quantile_loss
import dynamics_toolbox.constants.losses as losses


class QuantileModel(AbstractPlModel):
    """Network that outputs the full quantile distribution."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        encoder_cfg: DictConfig,
        loss_function: str = losses.CALI,
        num_quantile_draws: int = 30,
        learning_rate: float = 1e-3,
        sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        """
        Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            encoder_cfg: The configuration for the encoder.
            loss_function: The loss function to use, one of
                ["calibration", "pinball", "interval"]
            num_quantile_draws: Number of quantiles to randomly draw to compute loss.
            learning_rate: The learning rate for the network.
            hidden_activation: Activation of the networks hidden layers.
            sample_mode: The method to use for sampling.
            weight_decay: The weight decay for the optimizer.
    """
        super().__init__(input_dim, output_dim)
        self.save_hyperparameters()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._quantile_network = hydra.utils.instantiate(
            encoder_cfg,
            input_dim=input_dim+1,
            output_dim=output_dim,
            _recursive=False,
        )
        # Get loss function and optimizer fields
        self._loss_function = get_quantile_loss(loss_function)
        self._num_quantile_draws = num_quantile_draws
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = sample_mode
        self._kwargs = kwargs
        self._fixed_q_list = self._kwargs.get("fixed_q_list", None)

    def get_q_list(self) -> torch.Tensor:
        """Get a flat tensor of quantile levels.

        Returns: The fixed list of quantiles if it exists, else a uniform sample.
        """
        if self._fixed_q_list is not None:
            q_list = self._fixed_q_list
        else:
            q_list = torch.rand(self._num_quantile_draws)
        return q_list

    def forward(
        self,
        x: torch.Tensor,
        q_list: torch.Tensor = None,
        recal_model: Any = None,
        recal_type: str = None,
    ) -> torch.Tensor:
        """Get output for given list of quantiles

        Args:
            x: tensor, of size (num_x, dim_x)
            q_list: flat tensor of quantiles, if None, is set to [0.01, ..., 0.99]
            recal_model: a recalibration model #TODO
            recal_type: recalibration type  #TODO

        Returns:
            Given N input (x) points and K quantiles, outputs a NxK tensor
        """
        if q_list is None:
            q_list = torch.linspace(0.01, 0.99, 99)
        else:
            q_list = q_list.flatten()
        # handle recalibration
        if recal_model is not None:
            if recal_type == "torch":
                recal_model.cpu()  # keep recal model on cpu
                with torch.no_grad():
                    q_list = recal_model(q_list.reshape(-1, 1)).item().flatten()
            elif recal_type == "sklearn":
                q_list = recal_model.predict(q_list).flatten()
            else:
                raise ValueError("recal_type incorrect")
        num_pts = x.shape[0]
        num_q = q_list.shape[0]
        q_rep = q_list.view(-1, 1).repeat(1, num_pts).view(-1, 1)
        x_stacked = x.repeat(num_q, 1)
        model_in = torch.cat([x_stacked, q_rep], dim=1)  # note input is [x, p]
        q_pred = self._quantile_network(model_in).reshape(num_q, num_pts).T
        assert q_pred.shape == (num_pts, num_q)
        return q_pred

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """
        xi, _ = batch
        q_list = self.get_q_list()
        q_pred = self.forward(x=xi, q_list=q_list, recal_model=None, recal_type=None)
        return {'q_list': q_list, 'q_pred': q_pred}

    def get_eval_net_out(self, batch: Sequence[torch.Tensor]) \
            -> Dict[str, torch.Tensor]:
        """Get the validation output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """
        xi, _ = batch
        q_pred = self.forward(x=xi, q_list=None, recal_model=None, recal_type=None)
        return {
            'q_list': torch.linspace(0.01, 0.99, 99),
            'q_pred': q_pred,
            'median': q_pred[:, [49]]
        }

    def loss(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network, which for this model will be empty dict.
            batch: The batch passed into the network.

        Returns:
            The loss and a dictionary of other statistics.
        """
        xi, yi = batch
        q_list = net_out['q_list']
        q_pred = net_out['q_pred']
        return self._compute_quantile_model_loss_stats(
            xi=xi, yi=yi, q_pred=q_pred, q_list=q_list
        )

    def _compute_quantile_model_loss_stats(
            self,
            xi: torch.Tensor,
            yi: torch.Tensor,
            q_pred: torch.Tensor,
            q_list: torch.Tensor,
        ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Given a list of quantile levels and a batch of data, compute the loss, stats.

        Args:
            xi: A torch tensor for the input data.
            yi: A torch tensor for the labels.
            q_list: A flat tensor of quantile levels.

        Returns: The loss and a dictionary of other statistics.
        """
        loss = self._loss_function(
            y=yi,
            q_pred=q_pred,
            q_list=q_list,
            args=self._kwargs,
        )
        median_pred = self.forward(x=xi, q_list=torch.Tensor([0.5]))
        mse = torch.mean((median_pred - yi) ** 2)
        #TODO: add in more quantities...like what though?
        stats = dict(
            quantile_model_loss=loss.item(),
            mse=mse.item()
        )
        stats['loss'] = loss.item()
        return loss, stats

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
        with torch.no_grad():
            q_list = torch.cat([torch.rand(size=(1,)), torch.Tensor[0]])
            quantile_preds = self.forward(x=net_in, q_list=q_list)
        info = {
            'median_deltas': quantile_preds[:, -1],
            'sampled_quantile': q_list[0]
        }
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            deltas = quantile_preds[:, 0]
        else:
            deltas = quantile_preds[:, -1]
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
        return self.single_sample_output_from_torch(net_in)

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        if self._sample_mode not in [sampling_modes.SAMPLE_FROM_DIST,
                                     sampling_modes.RETURN_MEDIAN]:
            raise ValueError(
                f'QuantileModel sample mode must either be {sampling_modes.SAMPLE_FROM_DIST} '
                f'or {sampling_modes.RETURN_MEDIAN}, but received {mode}.')
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
        return {}

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
            {'prediction': net_out['median']},
            batch,
        )

