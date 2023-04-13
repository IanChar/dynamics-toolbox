"""
Abstract class for replay buffer.
"""
import abc
from typing import Dict

import numpy as np


class ReplayBuffer(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def clear_buffer(self):
        """Clear the buffer."""

    @abc.abstractmethod
    def add_paths(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        rews: np.ndarray,
        terminals: np.ndarray,
        **kwargs
    ):
        """Add paths taken in the environment.

        Args:
            obs: The observations with shape (num_paths, horizon + 1, obs_dim)
                or (horizon + 1, obs_dim).
            acts: The actions with shape (num_paths, horizon, act_dim)
                or (horizon, act_dim).
            rews: The rewards with shape (num_paths, horizon, 1)
                or (horizon, 1).
            terminals: The terminals with shape (num_paths, horizon, 1)
                or (horizon, 1).
        """

    @abc.abstractmethod
    def sample_batch(self, num_samples: int) -> Dict[str, np.ndarray]:
        """Sample a batch from the buffer."""

    @abc.abstractmethod
    def sample_starts(self, num_samples: int) -> np.ndarray:
        """Sample a batch from the buffer."""

    def end_epoch(self):
        pass
