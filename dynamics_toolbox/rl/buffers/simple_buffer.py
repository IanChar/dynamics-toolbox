"""
A simple replay buffer.

Author: Ian Char
Date: April 6, 2023
"""
from typing import Dict, Union

import numpy as np

from dynamics_toolbox.rl.buffers.abstract_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_buffer_size: int,
    ):
        """Constructor.

        Args:
            obs_dim: Dimension of the observation space.
            act_dim: Dimension of the action space.
            max_buffer_size: The maximum buffer size.
        """
        self._obs = np.zeros((max_buffer_size, obs_dim))
        self._next_obs = np.zeros((max_buffer_size, obs_dim))
        self._acts = np.zeros((max_buffer_size, act_dim))
        self._rews = np.zeros((max_buffer_size, 1))
        self._terms = np.zeros((max_buffer_size, 1), dtype='uint8')
        self._ptr = 0
        self._size = 0
        self._max_size = max_buffer_size

    def add_paths(
        self,
        obs: np.ndarray,
        acts: np.ndarray,
        rews: np.ndarray,
        terminals: np.ndarray,
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
        # Restructure the data
        if len(obs.shape) > 2:
            curr_obs = obs[:, :-1].reshape(-1, obs.shape[-1])
            nxt_obs = obs[:, 1:].reshape(-1, obs.shape[-1])
        else:
            curr_obs = obs[:-1]
            nxt_obs = obs[1:]
        acts, rews, terminals = [d.reshape(-1, d.shape[-1])
                                 for d in (acts, rews, terminals)]
        # Figure out if we will wrap around on the buffer.
        num2add = len(curr_obs)
        if self._max_size - self._ptr < num2add:
            from_ptr = self._max_size - self._ptr
            from_bottom = num2add - from_ptr
        else:
            from_ptr = num2add
            from_bottom = 0
        # Add the data to the buffer.
        for buffer, data in (
                (self._obs, curr_obs),
                (self._next_obs, nxt_obs),
                (self._acts, acts),
                (self._rews, rews),
                (self._terms, terminals)):
            buffer[self._ptr:self._ptr + from_ptr] = data[:from_ptr]
            if from_bottom > 0:
                buffer[:from_bottom] = data[-from_bottom:]
        # Update the information about the buffers.
        self._ptr = (self._ptr + num2add) % self._max_size
        if self._size < self._max_size:
            self._size = min(num2add + self._size, self._max_size)

    def add_step(
        self,
        obs: np.ndarray,
        nxt: np.ndarray,
        act: np.ndarray,
        rew: Union[float, np.ndarray],
        terminal: Union[bool, np.ndarray],
    ):
        """Add paths taken in the environment.

        Args:
            obs: The observations with shape (obs_dim,)
            nxt: The next observations with shape (obs_dim,)
            act: The actions with shape (act_dim,)
            rew: The rewards as a float preferabbly..
            terminal: The terminal as bool preferably.
        """
        rew = float(rew)
        terminal = bool(terminal)
        # Add to the buffer.
        for buffer, data in (
                (self._obs, obs),
                (self._next_obs, nxt),
                (self._acts, act),
                (self._rews, rew),
                (self._terms, terminal)):
            buffer[self._ptr] = data
        # Update the information about the buffers.
        self._ptr = (self._ptr + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def sample_batch(self, num_samples: int) -> Dict[str, np.ndarray]:
        """Sample a batch from the buffer.

        Args:
            num_samples: Number of samples for the batch.

        Returns:
            obs: Observaitons shape (batch_size, obs_dim).
            acts: Actions shape (batch_size, act_dim).
            rews: Rewards shape (batch_size, 1).
            nxts: Next Observations shape (batch_size, obs_dim).
            terms: Terminals shape (batch_size, 1).
        """
        idxs = np.random.randint(0, self._size, size=num_samples)
        return {
            'obs': self._obs[idxs],
            'acts': self._acts[idxs],
            'rews': self._rews[idxs],
            'nxts': self._next_obs[idxs],
            'terms': self._terms[idxs],
        }

    def sample_starts(self, num_samples: int) -> np.ndarray:
        """Sample observations to be used as starts from the buffer

        Args:
            num_samples: Number of samples for the batch.

        Returns:
            obs: Observaitons shape (batch_size, obs_dim).
        """
        idxs = np.random.randint(0, self._size, size=num_samples)
        return self._obs[idxs]
