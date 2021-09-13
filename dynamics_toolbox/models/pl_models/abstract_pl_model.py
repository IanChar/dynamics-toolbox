"""
Abstract deep model for learning dynamics.

Author: Ian Char
"""
import abc
from argparse import ArgumentParser
from typing import Dict, Tuple, NoReturn, Sequence, Any

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.argparse import add_argparse_args

from dynamics_toolbox.models.abstract_dynamics_model import AbstractDynamicsModel


class AbstractPlModel(LightningModule, AbstractDynamicsModel, metaclass=abc.ABCMeta):
    """Abstract model for predicting next states in dynamics."""

    def training_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for pytorch lightning. Returns the loss."""
        net_out = self.get_net_out(batch)
        loss, loss_dict = self.loss(net_out, batch)
        self._log_stats(loss_dict, prefix='train')
        return loss

    def validation_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> NoReturn:
        """Training step for pytorch lightning. Returns the loss."""
        net_out = self.get_net_out(batch)
        loss, loss_dict = self.loss(net_out, batch)
        loss_dict.update(self._get_test_and_validation_metrics(net_out, batch))
        self._log_stats(loss_dict, prefix='val')

    def test_step(self, batch: Sequence[torch.Tensor], batch_idx: int) -> NoReturn:
        """Training step for pytorch lightning. Returns the loss."""
        net_out = self.get_net_out(batch)
        loss, loss_dict = self.loss(net_out, batch)
        loss_dict.update(self._get_test_and_validation_metrics(net_out, batch))
        self._log_stats(loss_dict, prefix='test')

    def predict(self, states: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Predict the next state given current state and action.
        Args:
            states: The current states as a torch tensor.
            actions: The actions to be played as a torch tensor.
        Returns: The output of the model and give a dictionary of related quantities.
        """
        if isinstance(actions, int) or isinstance(actions, float):
            actions = np.array([actions]).reshape(1, 1)
        pt_states = torch.Tensor(states).to(self.device)
        pt_actions = torch.Tensor(actions).to(self.device)
        if len(states.shape) == 1:
            pt_states = pt_states.unsqueeze(0)
        if len(actions.shape) == 1:
            pt_actions = pt_actions.unsqueeze(int(pt_states.shape[0] != 1))
        net_in = torch.cat([pt_states, pt_actions], dim=1)
        deltas, infos = self._get_output_from_torch(net_in)
        if len(states.shape) == 1:
            deltas = deltas.flatten()
        return deltas.cpu().numpy(), infos

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

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        """Extend argparse."""
        return add_argparse_args(cls, parent_parser, **kwargs)

    @abc.abstractmethod
    def _get_output_from_torch(self, net_in: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the delta in state

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """

    def _log_stats(self, *args: Dict[str, float], prefix='train', **kwargs) -> NoReturn:
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
        return {}
