"""
A simple replay buffer.

Author: Ian Char
Date: April 6, 2023
"""
from typing import Dict, Tuple, Union

import numpy as np

from dynamics_toolbox.data.pl_data_modules.forward_dynamics_data_module import (
    ForwardDynamicsDataModule,
)
from dynamics_toolbox.rl.buffers.abstract_buffer import ReplayBuffer


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_buffer_size: int,
        clear_every_n_epochs: int = -1,
    ):
        """Constructor.

        Args:
            obs_dim: Dimension of the observation space.
            act_dim: Dimension of the action space.
            max_buffer_size: The maximum buffer size.
            clear_every_n_epoch: Whether to clear the buffer after every epoch.
        """
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        max_buffer_size = int(max_buffer_size)
        self._max_size = max_buffer_size
        self._clear_every_n_epochs = clear_every_n_epochs
        self._countdown_to_clear = (float('inf') if clear_every_n_epochs < 1
                                    else clear_every_n_epochs)
        self.clear_buffer()

    def clear_buffer(self):
        """Clear the buffer."""
        self._obs = np.zeros((self._max_size, self._obs_dim))
        self._next_obs = np.zeros((self._max_size, self._obs_dim))
        self._acts = np.zeros((self._max_size, self._act_dim))
        self._rews = np.zeros((self._max_size, 1))
        self._terms = np.zeros((self._max_size, 1), dtype='uint8')
        self._ptr = 0
        self._size = 0

    def add_paths(self, paths: Dict[str, np.ndarray]):
        """Add paths taken in the environment.

        Args:
            Dict with...
            obs: The observations with shape (num_paths, horizon + 1, obs_dim)
                or (horizon + 1, obs_dim).
            acts: The actions with shape (num_paths, horizon, act_dim)
                or (horizon, act_dim).
            rews: The rewards with shape (num_paths, horizon, 1)
                or (horizon, 1).
            terms: The terminals with shape (num_paths, horizon, 1)
                or (horizon, 1).
            Optionaly masks: shape (num_paths, horizon, 1) or (horizon, 1).
        """
        obs, acts, rews, terms = [paths[k] for k in ('observations', 'actions',
                                                     'rewards', 'terminals')]
        # Restructure the data
        if len(obs.shape) > 2:
            curr_obs = obs[:, :-1].reshape(-1, obs.shape[-1])
            nxt_obs = obs[:, 1:].reshape(-1, obs.shape[-1])
        else:
            curr_obs = obs[:-1]
            nxt_obs = obs[1:]
        acts, rews, terms = [d.reshape(-1, d.shape[-1])
                             for d in (acts, rews, terms)]
        if 'masks' in paths:
            masks = np.argwhere(paths['masks'].flatten()).flatten()
            curr_obs, nxt_obs, acts, rews, terms = [
                d[masks] for d in (curr_obs, nxt_obs, acts, rews, terms)]
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
                (self._terms, terms)):
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
            'observations': self._obs[idxs],
            'actions': self._acts[idxs],
            'rewards': self._rews[idxs],
            'next_observations': self._next_obs[idxs],
            'terminals': self._terms[idxs],
        }

    def sample_starts(self, num_samples: int) -> Tuple[np.ndarray, Dict]:
        """Sample observations to be used as starts from the buffer

        Args:
            num_samples: Number of samples for the batch.

        Returns:
            obs: Observaitons shape (batch_size, obs_dim).
        """
        idxs = np.random.randint(0, self._size, size=num_samples)
        return self._obs[idxs], {}

    def to_forward_dynamics_module(
        self,
        **kwargs
    ) -> ForwardDynamicsDataModule:
        """Conver the current buffer to a forward dynamics module.."""
        return ForwardDynamicsDataModule(
            data_source='N/A',
            qset={
                'observations': self._obs[:self._size],
                'actions': self._acts[:self._size],
                'rewards': self._rews[:self._size],
                'next_observations': self._next_obs[:self._size],
                'terminals': self._terms[:self._size]
            },
            **kwargs
        )

    def end_epoch(self):
        self._countdown_to_clear -= 1
        if self._countdown_to_clear <= 0:
            self.clear_buffer()
            self._countdown_to_clear = self._clear_every_n_epochs


class SimpleOfflineReplayBuffer(SimpleReplayBuffer):

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        **kwargs
    ):
        """Constructor.

        Args:
            data with observations, next_observations, rewards, actions, terminals.
        """
        self._obs = data['observations']
        self._next_obs = data['next_observations']
        self._acts = data['actions']
        self._rews = data['rewards'].reshape(-1, 1)
        self._terms = data['terminals'].reshape(-1, 1)
        max_buffer_size = len(self._obs)
        self._ptr = 0
        self._size = max_buffer_size
        self._max_size = max_buffer_size
        self._clear_every_n_epochs = float('inf')
        self._countdown_to_clear = float('inf')
