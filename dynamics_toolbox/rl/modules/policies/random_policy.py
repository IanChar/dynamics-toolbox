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
        action_dim: int = None,
        action_space=None,
    ):
        """Constructor.

        Args:
            action_dim: The dimension of the action space.
            action_space: The action space to sample from if provided.
            action_plan: The actions in the plan to take. Should have shape
                (num_plans, horizon, act_dim)
        """
        assert not (action_dim is None and action_space is None)
        if action_space is None:
            self.action_dim = action_dim
        else:
            self.action_dim = len(action_space.low)
        self.action_space = action_space

    def get_actions(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get multiple actions.

        Args:
            obs_np: numpy array of shape (batch_size, obs_dim)

        Returns:
            * Sampled actions w shape (batch_size, act_dim)
            * Logprobabilities for the actions (batch_size,)
        """
        if len(obs_np.shape) == 1:
            obs_np = obs_np[np.newaxis]
        if self.action_space is None:
            acts = np.random.uniform(-1, 1, size=(len(obs_np, self.action_dim)))
        acts = np.array([self.action_space.sample() for _ in range(len(obs_np))])
        return np.squeeze(acts, axis=0), np.ones(len(acts))

    @property
    def act_dim(self) -> int:
        """Action dimension."""
        return self.action_dim
