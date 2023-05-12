"""
Bounding on dynamics output.

Author: Ian Char
Date: May 11, 2023
"""
from typing import Dict, Optional

import numpy as np


class Bounder:

    def __init__(
        self,
        state_lower_bounds: Optional[np.ndarray] = None,
        state_upper_bounds: Optional[np.ndarray] = None,
        velocity_lower_bounds: Optional[np.ndarray] = None,
        velocity_upper_bounds: Optional[np.ndarray] = None,
        reward_lower_bounds: Optional[np.ndarray] = None,
        reward_upper_bounds: Optional[np.ndarray] = None,
    ):
        """Constructor.

        Args:
            state_lower_bounds: Lower bounds on states.
            state_upper_bounds: Upper bounds on states.
            velocity_lower_bounds: Lower bounds on velocity in state.
            velocity_upper_bounds: Upper bounds on velocity in state.
            reward_lower_bounds: Lower bounds on reward.
            reward_upper_bounds: Upper bounds on reward.
        """
        self._state_lower = state_lower_bounds
        self._state_upper = state_upper_bounds
        self._vel_lower = velocity_lower_bounds
        self._vel_upper = velocity_upper_bounds
        self._rew_lower = reward_lower_bounds
        self._rew_upper = reward_upper_bounds

    def bound_state(
        self,
        state: np.ndarray,
        prev: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Bound the state.

        Args:
            state: The state to bound.
            prev: Previous state. If this is not provided, will not bound by velocity.

        Returns: The bounded state.
        """
        clipped = state
        if self._state_lower is not None and self._state_upper is not None:
            clipped = np.clip(state, self._state_lower, self._state_upper)
        if (prev is not None and self._vel_lower is not None 
                and self._vel_upper is not None):
            clipped = np.clip(
                clipped,
                prev + self._vel_lower,
                prev + self._vel_upper,
            )
        return clipped

    def bound_reward(
        self,
        reward: np.ndarray,
    ) -> np.ndarray:
        """Bound the reward.

        Args:
            reward: The reward to be clipped.

        Returns: Clipped reward.
        """
        if self._rew_lower is not None and self._rew_upper is not None:
            return np.clip(reward, self._rew_lower, self._rew_upper)
        return reward

    @classmethod
    def bound_from_dataset(
        cls,
        data: Dict[str, np.ndarray],
        spread_amt: float,
        bound_state: bool = True,
        bound_velocity: bool = True,
        bound_reward: bool = True,
        account_for_d4rl_bug: bool = True,
    ):
        """Instantiate from a dataset.

        Args:
            data: Dictionary of the data. Should contain observations, rewards,
                and next_observation.
            spread_amt: Amount of spread. e.g. 1.1 will be min and max in the dataset
                plus an addition 10% of the spread between them on both sides.
            bound_state: Whether state should be bounded.
            bound_velocity: Whether velocity should be bounded.
            bound_reward: Whether reward should be bounded.
            account_for_d4rl_bug: This accounts for the fact that transitions are
                bogus when terminals are present.
        """
        if bound_state:
            state_mins = np.min(data['observations'], axis=0)
            state_maxs = np.max(data['observations'], axis=0)
            spread = state_maxs - state_mins
            midpt = (state_maxs + state_mins) / 2
            state_lows = midpt - spread / 2 * spread_amt
            state_highs = midpt + spread / 2 * spread_amt
        else:
            state_lows, state_highs = None, None
        if bound_velocity:
            if account_for_d4rl_bug and 'terminals' in data:
                valid_idxs = np.argwhere(1 - data['terminals']).flatten()
                vels = (data['next_observations'][valid_idxs]
                        - data['observations'][valid_idxs])
            else:
                vels = data['next_observations'] - data['observations']
            vel_mins = np.min(vels, axis=0)
            vel_maxs = np.max(vels, axis=0)
            spread = vel_maxs - vel_mins
            midpt = (vel_maxs + vel_mins) / 2
            vel_lows = midpt - spread / 2 * spread_amt
            vel_highs = midpt + spread / 2 * spread_amt
        else:
            vel_lows, vel_highs = None, None
        if bound_reward:
            rew_mins = np.min(data['rewards'], axis=0)
            rew_maxs = np.max(data['rewards'], axis=0)
            spread = rew_maxs - rew_mins
            midpt = (rew_maxs + rew_mins) / 2
            rew_lows = midpt - spread / 2 * spread_amt
            rew_highs = midpt + spread / 2 * spread_amt
        else:
            rew_lows, rew_highs = None, None
        return cls(state_lows, state_highs, vel_lows, vel_highs, rew_lows, rew_highs)
