"""
Abstract class for models that are train on sequences of SARSA data.

Author: Ian Char
"""
import abc
from typing import Dict, Sequence

import torch

from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel


class AbstractSequentialModel(AbstractPlModel, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def warm_up_period(self) -> int:
        """Amount of data to take in before starting to predict"""

    @property
    @abc.abstractmethod
    def record_history(self) -> bool:
        """Whether to keep track of the quantities being fed into the neural net."""

    @record_history.setter
    @abc.abstractmethod
    def record_history(self, mode: bool) -> None:
        """Set whether to keep track of quantities being fed into the neural net."""

    def clear_history(self) -> None:
        """Clear the history."""
        pass

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()

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
        one_step_pred = pred[:, self._warm_up_period, ...]
        pred = pred.reshape(-1, pred.shape[-1])
        yi = batch[3]
        one_step_yi = yi[:, self._warm_up_period, ...]
        yi = yi.reshape(-1, yi.shape[-1])
        for metric_name, metric in self.metrics.items():
            metric_value = metric(one_step_pred, one_step_yi)
            if len(metric_value.shape) > 0:
                for dim_idx, metric_v in enumerate(metric_value):
                    to_return[f'{metric_name}_dim{dim_idx}_OneStep'] = metric_v
            else:
                to_return[f'{metric_name}_OneStep'] = metric_value
        for metric_name, metric in self.metrics.items():
            metric_value = metric(pred, yi)
            if len(metric_value.shape) > 0:
                for dim_idx, metric_v in enumerate(metric_value):
                    to_return[f'{metric_name}_dim{dim_idx}'] = metric_v
            else:
                to_return[metric_name] = metric_value
        return to_return
