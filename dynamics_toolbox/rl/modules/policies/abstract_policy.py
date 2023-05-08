"""
Abstract policy class.

Auhtor: Ian Char
Date: April 6, 2023
"""
import abc
from typing import Tuple, Union

import numpy as np


class Policy(metaclass=abc.ABCMeta):

    def reset(self):
        pass

    def train(self, mode: bool = True):
        pass

    def eval(self):
        self.train(False)

    def get_reward_feedback(self, rewards: Union[float, np.ndarray]):
        """Get feedback from the environment about the last reward.

        Args:
            rewards: The rewards as a float or list of rewards if doing multiple
                     rollouts.
        """
        pass

    @abc.abstractmethod
    def get_actions(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get multiple actions.

        Args:
            obs_np: numpy array of shape (batch_size, obs_dim)

        Returns:
            * Sampled actions w shape (batch_size, act_dim)
            * Logprobabilities for the actions (batch_size,)
        """

    @property
    @abc.abstractmethod
    def act_dim(self) -> int:
        """Action dimension."""
