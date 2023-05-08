"""
Utility functions for wrapping models into environments.

Author: Ian Char
Date: April 5, 2023
"""
from typing import Callable, Dict

import gym
import numpy as np


###########################################################################
#                            START DIST FUNCTIONS                         #
###########################################################################

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


def start_state_dist_from_data(
    data: Dict[str, np.ndarray],
) -> Callable[[int], np.ndarray]:
    """Create start state distribution from some dataset.

    Args:
        data: Dictionary of data. Only requirement is to have observations in data.

    Returns: Function that given a number of start states returns those start states
        as an ndarray with shape (num starts, obs dim).
    """
    assert 'observations' in data, 'Require observations key to be in data.'
    obs_data = data['observations']

    def start_state_dist(num_starts: int):
        return data['observations'][np.random.randint(0,
                                                      len(obs_data), size=num_starts)]

    return start_state_dist

###########################################################################
#                            TERMINAL FUNCTIONS                           #
###########################################################################


def get_terminal_from_env_name(env_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Get terminal function from the name of the environment.

    Args:
        env_name: Name of the environment.

    Returns: terminals function mapping states to terminal values.
    """
    if 'hopper' in env_name.lower():
        return hopper_terminal
    if 'walker' in env_name.lower():
        return walker_terminal
    return no_terminal


def no_terminal(states: np.ndarray) -> np.ndarray:
    """No terminal

    Args:
        states: Current states w shape (batch_size, obs_dim).

    Returns: ndarray containing if it is a terminal state or not (batch_size,)
    """
    states = _add_axis_if_needed(states)
    return np.full(states.shape[0], False)


def hopper_terminal(states: np.ndarray) -> np.ndarray:
    """Terminal function for hopper. Taken from the MOPO repo

    https://github.com/tianheyu927/mopo

    Args:
        states: Current states w shape (batch_size, obs_dim).

    Returns: ndarray containing if it is a terminal state or not (batch_size,)
    """
    states = _add_axis_if_needed(states)
    height = states[:, 0]
    angle = states[:, 1]
    return np.logical_or.reduce([
        ~np.isfinite(states).all(axis=-1),
        np.abs(states[:, 1:] >= 100).all(axis=-1),
        height <= 0.7,
        np.abs(angle) >= 0.2,
    ])


def walker_terminal(states: np.ndarray) -> np.ndarray:
    """Terminal function for walker. Taken from the MOPO repo

    https://github.com/tianheyu927/mopo

    Args:
        states: Current states w shape (batch_size, obs_dim).

    Returns: ndarray containing if it is a terminal state or not (batch_size,)
    """
    states = _add_axis_if_needed(states)
    height = states[:, 0]
    angle = states[:, 1]
    return np.logical_or.reduce([
        height <= 0.8,
        height >= 2.0,
        angle <= -1.0,
        angle >= 1.0,
    ])


def _add_axis_if_needed(states):
    if len(states.shape) == 1:
        states = states[np.newaxis, ...]
    return states
