"""
Policy that gives random actions.

Author: Ian Char
Date: April 6, 2023
"""
from typing import Tuple

import numpy as np

from dynamics_toolbox.rl.modules.policies.abstract_policy import Policy


class RandomPolicy(Policy):

    def __init__(
        self,
        action_dim: int,
        action_space=None,
    ):
        """Constructor.

        Args:
            action_dim: The dimension of the action space.
            action_space: The action space to sample from if provided.
            action_plan: The actions in the plan to take. Should have shape
                (num_plans, horizon, act_dim)
        """
        self.action_dim = action_dim
        self.action_space = action_space

    def get_action(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get action.

        Args:
            obs_np: numpy array of shape (obs_dim,)

        Returns:
            * Sampled actions.
            * Log probability.
        """
        return self.get_actions(obs_np[np.newaxis]).flatten()

    def get_actions(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get multiple actions.

        Args:
            obs_np: numpy array of shape (batch_size, obs_dim)

        Returns:
            * Sampled actions w shape (batch_size, act_dim)
            * Logprobabilities for the actions (batch_size,)
        """
        if self.action_space is None:
            return np.random.uniform(-1, 1, size=(len(obs_np, self.action_dim)))
        return self.action_space.sample(len(obs_np))

    @property
    def act_dim(self) -> int:
        """Action dimension."""
        return self.action_dim
