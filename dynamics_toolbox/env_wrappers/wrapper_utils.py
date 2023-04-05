"""
Utility functions for wrapping models into environments.

Author: Ian Char
Date: April 5, 2023
"""
from typing import Callable

import gym
import numpy as np


def start_state_dist_from_env(env: gym.Env) -> Callable[[int], np.ndarray]:
    """Create a start state distribution by repeatedly calling an environments reset.

    Args:
        env: The environment to wrap.

    Returns: Function that given a number of start states returns those start states
        as an ndarray with shape (num starts, obs dim).
    """

    def start_state_dist(num_starts: int):
        return np.array([env.reset() for _ in range(num_starts)])

    return start_state_dist
