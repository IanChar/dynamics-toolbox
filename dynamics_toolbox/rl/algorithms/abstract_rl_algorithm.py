"""
Abstract class for RL algorithm. An RL algorithm consists of losses and how to
update the networks given a batch of data.

Author: Ian Char
Date: April 6, 2023
"""
import abc
from typing import Dict, Tuple

import numpy as np
from torch import Tensor

from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm


class RLAlgorithm(metaclass=abc.ABCMeta):

    def grad_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Do a gradient step.

        Args:
            batch: Dictionary of fields w shape (batch_size, *)

        Returns: Dictionary of loss statistics.
        """
        pt_batch = {k: dm.torch_ify(v) for k, v in batch.items()}
        self.policy.deterministic = False
        self.policy.train()
        losses, loss_stats = self._compute_losses(pt_batch)
        # Do updates to both policy and value.
        self.policy_optimizer.zero_grad()
        losses['policy_loss'].backward()
        self.policy_optimizer.step()
        self.val_optimizer.zero_grad()
        losses['val_loss'].backward()
        self.val_optimizer.step()
        self._post_grad_step_updates()
        return loss_stats

    def _post_grad_step_updates(self):
        pass

    @abc.abstractmethod
    def _compute_losses(self, pt_batch: Dict[str, Tensor]) -> Tuple[Dict[str, float]]:
        """Compute the loasses.

        Args:
            pt_batch: Dictionary of fields w shape (batch_size, *)

        Returns: Dictionary of loss statistics and dicionary of losses.
        """

    @property
    @abc.abstractmethod
    def policy_optimizer(self):
        """Optimzer for policy."""

    @property
    @abc.abstractmethod
    def val_optimizer(self):
        """Optimzier for value functions."""

    @property
    @abc.abstractmethod
    def policy(self):
        """The policy."""
