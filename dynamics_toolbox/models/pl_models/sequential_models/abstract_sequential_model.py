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
        if 'prediction' in net_out:
            pred = net_out['prediction']
        elif 'y_i' in net_out:
            pred = net_out['y_i']
        elif 'mean' in net_out:
            pred = net_out['mean']
        else:
            raise ValueError('Need either prediction or mean in the net_out')
        if len(batch) > 3:  # Check if we are doing RL data or (x, y, mask) data.
            raise NotImplementedError('RL data needs to be reimplemented.')
        yi, mask = batch[1:]
        for metric_name, metric in self.metrics.items():
            metric_value = metric(pred, yi, mask)
            if len(metric_value.shape) > 0:
                for dim_idx, metric_v in enumerate(metric_value):
                    to_return[f'{metric_name}_{self._dim_name_map[dim_idx]}'] = metric_v
            else:
                to_return[metric_name] = metric_value
        return to_return
