"""
Abstract class for RL algorithm. An RL algorithm consists of losses and how to
update the networks given a batch of data.

Author: Ian Char
Date: April 6, 2023
"""
import abc
from typing import Dict

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
        loss, loss_stats = self._compute_losses(pt_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss_stats

    @abc.abstractmethod
    def _compute_losses(self, pt_batch: Dict[str, Tensor]) -> Dict[str, float]:
        """Compute the loasses.

        Args:
            pt_batch: Dictionary of fields w shape (batch_size, *)

        Returns: Dictionary of loss statistics.
        """

    @property
    @abc.abstractmethod
    def optimizer(self):
        """Optimzier."""

    @property
    @abc.abstractmethod
    def policy(self):
        """The policy."""
