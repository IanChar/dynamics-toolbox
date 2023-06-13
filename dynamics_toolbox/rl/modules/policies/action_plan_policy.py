"""
Policy that wraps aaction plan.

Author: Ian Char
Date: April 6, 2023
"""
from typing import Tuple

import numpy as np

from dynamics_toolbox.rl.modules.policies.abstract_policy import Policy


class ActionPlanPolicy(Policy):

    def __init__(
        self,
        action_plan: np.ndarray,
    ):
        """Constructor.

        Args:
            action_plan: The actions in the plan to take. Should have shape
                (num_plans, horizon, act_dim)
        """
        self._t = 0
        self._action_plan = action_plan
        self._max_horizon = action_plan.shape[1]

    def reset(self):
        self._t = 0

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
        if len(obs_np) != len(self._action_plan):
            raise ValueError('Expected observations to match number of plans')
        if self._t >= self._max_horizon:
            raise RuntimeError(f'Plan only valid for {self._max_horizon} timesteps.')
        self._t += 1
        return (
            np.squeeze(self._action_plan[:, self._t - 1], axis=0),
            np.ones(len(obs_np)),
        )

    @property
    def act_dim(self) -> int:
        """Action dimension."""
        return self._action_plan.shape[-1]
