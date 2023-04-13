"""
A replay buffer that holds sequential information.

NOTE: This is a really bad way of implementing it. I basically take simple replay
buffer and multiply it by lookback size. A lot of unforseen difficulties come up
if you try to do it any smarter way, especially if dealing with an episode that
terminates early.

Author: Ian Char
Date: April 13, 2023
"""
from typing import Dict, Union

import numpy as np

from dynamics_toolbox.rl.buffers.abstract_buffer import ReplayBuffer


class SequenceReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_buffer_size: int,
        lookback: int,
    ):
        """
        Constructor.

        Args:
            obs_dim: Dimension of the observation space.
            act_dim: Dimension of the action space.
            max_buffer_size: The maximum buffer size.
            lookback: How big the lookback should be.
        """
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._max_size = int(max_buffer_size)
        self._lookback = lookback
        self.clear_buffer()

    def clear_buffer(self):
        """Clear all of the buffers."""
        self._observations = np.zeros((
            self._max_size,
            self._lookback,
            self._obs_dim,
        ))
        self._next_observations = np.zeros((
            self._max_size,
            self._lookback,
            self._obs_dim,
        ))
        # NOTE: actions and rewards have one padding at the beginning. This is because
        # when encoding for the first time step we need to encode previous actions
        # and rewards but there are none at that point.
        self._actions = np.zeros((
            self._max_size,
            self._lookback + 1,
            self._act_dim,
        ))
        self._rewards = np.zeros((
            self._max_size,
            self._lookback + 1,
            1,
        ))
        self._terminals = np.zeros((
            self._max_size,
            self._lookback,
            1,
        ), dtype='uint8')
        self._masks = np.zeros((
            self._max_size,
            self._lookback,
            1,
        ), dtype='uint8')
        self._top = 0
        self._size = 0

    def add_paths(self, paths: Dict[str, np.ndarray]):
        """Add paths taken in the environment.

        TODO: This could be optimized by removing the nested for loops and adding
              multiple paths at once. More logic is needed though.

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
        if len(paths['actions'].shape) < 3:
            paths = {k: v[np.newaxis] for k, v in paths.items()}
        for pidx in range(len(paths['actions'])):
            path = {k: v[pidx] for k, v in paths.items}
            length = len(path['actions'])
            for strt in range(0, max(length - self._lookback + 1, 1)):
                end = min(strt + self._lookback, strt + length)
                for k, buff in (('actions', self._actions), ('rewards', self._rewards)):
                    buff[self._top, 1:end - strt + 1] = path[k][strt:end]
                for k, buff in (
                        ('observations', self._observations),
                        ('next_observations', self._next_observations),
                        ('terminals', self._terminals)):
                    buff[self._top, :end - strt] = path[k][strt:end]
                if 'masks' in path:
                    self._masks[self._top, :end - strt] = path['masks'][strt:end]
                else:
                    self._masks[self._top, :end - strt] = 1
                self._masks[self._top, end - strt:] = 0
                self._advance()

    def sample_batch(self, num_samples: int) -> Dict[str, np.ndarray]:
        """Get a random batch of data.

        Args:
            batch_size: number of sequences to grab.

        Returns: Dictionary of information for the update.
            observations: This is a history w shape (batch_size, L, obs_dim)
            actions: This is the history of actions (batch_size, L + 1, act_dim)
            rewards: This is the rewards at the last point (batch_size, L + 1, 1)
            next_observation: This is a history of nexts (batch_size, L, obs_dim)
            terminals: Whether last time step is terminals (batch_size, L, 1)
            masks: Masks of what is real and what data (batch_size, L, 1).
        """
        indices = np.random.randint(0, self._size, num_samples)
        batch = {}
        batch['observations'] = self._observations[indices]
        batch['next_observations'] = self._next_observations[indices]
        batch['actions'] = self._actions[indices]
        batch['rewards'] = self._rewards[indices]
        batch['terminals'] = self._terminals[indices]
        batch['masks'] = self._masks[indices]
        return batch

    def sample_starts(self, num_samples: int) -> np.ndarray:
        """Get a random batch of data.

        Args:
            batch_size: number of sequences to grab.

        Returns: Dictionary of information for the update.
            observations: This is a history w shape (batch_size, L, obs_dim)
            actions: This is the history of actions (batch_size, L + 1, act_dim)
            rewards: This is the rewards at the last point (batch_size, L + 1, 1)
            next_observation: This is a history of nexts (batch_size, L, obs_dim)
            terminals: Whether last time step is terminals (batch_size, L, 1)
            masks: Masks of what is real and what data (batch_size, L, 1).
        """
        indices = np.random.randint(0, self._size, num_samples)
        return self._observations[indices][:, -1]

    def add_step(
        self,
        obs: np.ndarray,
        nxt: np.ndarray,
        act: np.ndarray,
        rew: Union[float, np.ndarray],
        terminal: Union[bool, np.ndarray],
    ):
        """
        Add a transition tuple.
        """
        raise NotImplementedError('This would require a lot more logic :(')

    def _advance(self):
        self._top = (self._top + 1) % self._max_buffer_size
        if self._size < self._max_buffer_size:
            self._size += 1
