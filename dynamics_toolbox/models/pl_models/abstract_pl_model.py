"""
Abstract deep model for learning dynamics.

Author: Ian Char
"""
import abc
from typing import Dict, Tuple, Sequence, Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule

from dynamics_toolbox.models.abstract_model import AbstractModel


class AbstractPlModel(LightningModule, AbstractModel, metaclass=abc.ABCMeta):
    """Abstract model for predicting next states in dynamics."""

    def training_step(
            self,
            batch: Sequence[torch.Tensor],
            batch_idx: int,
    ) -> torch.Tensor:
        """Training step for pytorch lightning. Returns the loss."""
        net_out = self.get_net_out(batch)
        loss, loss_dict = self.loss(net_out, batch)
        self._log_stats(loss_dict, prefix='train')
        return loss

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> None:
        """Training step for pytorch lightning. Returns the loss."""
        net_out = self.get_net_out(batch)
        loss, loss_dict = self.val_loss(net_out, batch)
        loss_dict.update(self._get_test_and_validation_metrics(net_out, batch))
        self._log_stats(loss_dict, prefix='val')

    def test_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> None:
        """Training step for pytorch lightning. Returns the loss."""
        net_out = self.get_net_out(batch)
        loss, loss_dict = self.val_loss(net_out, batch)
        loss_dict.update(self._get_test_and_validation_metrics(net_out, batch))
        self._log_stats(loss_dict, prefix='test')

    def predict(
            self,
            model_input: np.ndarray,
            each_input_is_different_sample: Optional[bool] = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Make predictions using the currently set sampling method.

        Args:
            model_input: The input to be given to the model.
            each_input_is_different_sample: Whether each input should be treated
                as being drawn from a different sample of the model. Note that this
                may not have an effect on all models (e.g. PNN)

        Returns:
            The output of the model and give a dictionary of related quantities.
        """
        net_in = torch.Tensor(model_input).to(self.device)
        if each_input_is_different_sample:
            output, infos = self.multi_sample_output_from_torch(net_in)
        else:
            output, infos = self.single_sample_output_from_torch(net_in)
        return output.numpy(), infos

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer"""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @abc.abstractmethod
    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed to the network.

        Returns:
            Dictionary of name to tensor.
        """

    @abc.abstractmethod
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

    def val_loss(
            self,
            net_out: Dict[str, torch.Tensor],
            batch: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the validation loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network.

        Returns:
            The loss and a dictionary of other statistics.
        """
        return self.loss(net_out=net_out, batch=batch)

    @abc.abstractmethod
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

    @abc.abstractmethod
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

    @property
    @abc.abstractmethod
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        """Get the list of metric functions to compute."""

    @property
    @abc.abstractmethod
    def learning_rate(self) -> float:
        """Get the learning rate."""

    @property
    @abc.abstractmethod
    def weight_decay(self) -> float:
        """Get the weight decay."""

    def _log_stats(self, *args: Dict[str, float], prefix='train', **kwargs) -> None:
        """Log all of the stats from dictionaries.

        Args:
            args: Dictionaries of torch tensors to add stats about.
            prefix: The prefix to add to the statistic string.
            kwargs: Other kwargs to be passed to self.log.
        """
        for arg in args:
            for stat_name, stat in arg.items():
                self.log(f'{prefix}/{stat_name}', stat, **kwargs)

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
