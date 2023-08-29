"""
A replay buffer that holds sequential information.

NOTE: This is a really bad way of implementing it. I basically take simple replay
buffer and multiply it by lookback size. A lot of unforseen difficulties come up
if you try to do it any smarter way, especially if dealing with an episode that
terminates early.

Author: Ian Char
Date: April 13, 2023
"""
from typing import Dict, Union, Tuple, Optional

import numpy as np
import torch

from dynamics_toolbox.data.pl_data_modules.forward_dynamics_data_module import (
    ForwardDynamicsDataModule,
)
from dynamics_toolbox.rl.modules.history_encoders.abstract_history_encoder import (
    HistoryEncoder,
)
from dynamics_toolbox.rl.buffers.abstract_buffer import ReplayBuffer
from dynamics_toolbox.utils.sarsa_data_util import parse_into_trajectories
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm


class SequentialReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_buffer_size: int,
        lookback: int,
        clear_every_n_epochs: int = -1,
        encoding_dims: Optional[Dict[str, int]] = None
    ):
        """
        Constructor.

        Args:
            obs_dim: Dimension of the observation space.
            act_dim: Dimension of the action space.
            max_buffer_size: The maximum buffer size.
            lookback: How big the lookback should be.
            clear_every_n_epochs: Whether to clear the buffer after every epoch.
            q_encoding_dim: Optionally the replay buffer can also store encodings
                of the past up to this point.
        """
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._max_size = int(max_buffer_size)
        self._lookback = lookback
        self._clear_every_n_epochs = clear_every_n_epochs
        self.encoding_dims = encoding_dims
        self._countdown_to_clear = (float('inf') if clear_every_n_epochs < 1
                                    else clear_every_n_epochs)
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
        # Encodings are only saved for the start of the subsequence since we
        # expect the network to re-encode when training. In other words, the
        # buffers here are 2D since they have no lookback.
        if self.encoding_dims is not None:
            for k, v in self.encoding_dims.items():
                setattr(self, f'_{k}_encoding', np.zeros((
                    self._max_size,
                    v
                )))
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
            observations: The observations with shape (num_paths, horizon + 1, obs_dim)
                or (horizon + 1, obs_dim).
            actions: The actions with shape (num_paths, horizon, act_dim)
                or (horizon, act_dim).
            rewards: The rewards with shape (num_paths, horizon, 1)
                or (horizon, 1).
            terminals: The terminals with shape (num_paths, horizon, 1)
                or (horizon, 1).
            Optionaly
                masks: shape (num_paths, horizon, 1) or (horizon, 1).
                encodings: shape (num_paths, horizon, encoding_dim)
        """
        if len(paths['actions'].shape) < 3:
            paths = {k: v[np.newaxis] for k, v in paths.items()}
        paths['next_observations'] = paths['observations'][:, 1:]
        paths['observations'] = paths['observations'][:, :-1]
        for pidx in range(len(paths['actions'])):
            path = {}
            for k, v in paths.items():
                if k != 'info':
                    path[k] = v[pidx]
            length = None
            if 'masks' in path:
                idxs = np.argwhere(path['masks'] - 1).flatten()
                if len(idxs):
                    length = idxs[0]
            if length is None:
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
                # Possibly store encodings.
                for k, v in path.items():
                    if 'encoding' in k and hasattr(self, f'_{k}'):
                        getattr(self, f'_{k}')[self._top] = v[strt]
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
            next_observations: This is a history of nexts (batch_size, L, obs_dim)
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
        if self.encoding_dims is not None:
            for k in self.encoding_dims.keys():
                batch[f'{k}_encoding'] = getattr(self, f'_{k}_encoding')[indices]
        return batch

    def sample_starts(self, num_samples: int) -> np.ndarray:
        """Get a random batch of data.

        Args:
            batch_size: number of sequences to grab.

        Returns: Start states (num_samples, obs_dim)
        """
        # TODO: Currently this has a slight bug where we do not consider all of the
        # states for start states. Just the first state in the window.
        indices = np.random.randint(0, self._size, num_samples)
        return self._observations[indices][:, 0]

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
        self._top = (self._top + 1) % self._max_size
        if self._size < self._max_size:
            self._size += 1

    def to_forward_dynamics_module(
        self,
        **kwargs
    ) -> ForwardDynamicsDataModule:
        """Conver the current buffer to a forward dynamics module.."""
        raise NotImplementedError('TODO')

    def end_epoch(self):
        self._countdown_to_clear -= 1
        if self._countdown_to_clear <= 0:
            self.clear_buffer()
            self._countdown_to_clear = self._clear_every_n_epochs


class SequentialOfflineReplayBuffer(SequentialReplayBuffer):

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        lookback: int,
        **kwargs
    ):
        """Constructor.

        Args:
            data with observations, next_observations, rewards, actions, terminals.
        """
        encoding_dims = None
        for k, v in data.items():
            if 'encoding' in k:
                if encoding_dims is None:
                    encoding_dims = {}
                encoding_dims[k] = v.shape[-1]
        super().__init__(
            obs_dim=data['observations'].shape[-1],
            act_dim=data['actions'].shape[-1],
            max_buffer_size=len(data['actions']),
            lookback=lookback,
            encoding_dims=encoding_dims,
        )
        self._starts = data['observations']
        data['rewards'] = data['rewards'].reshape(-1, 1)
        data['terminals'] = data['terminals'].reshape(-1, 1)
        self._paths = parse_into_trajectories(data)
        self._clear_every_n_epochs = float('inf')
        self._countdown_to_clear = float('inf')
        for path in self._paths:
            path['observations'] = np.concatenate([
                path['observations'],
                path['next_observations'][[0]],
            ], axis=0)
            self.add_paths(path)

    def sample_starts(self, num_samples: int) -> Tuple[np.ndarray, Dict]:
        """Get a random batch of data.

        Args:
            batch_size: number of sequences to grab.

        Returns: Start states (num_samples, obs_dim)
        """
        indices = np.random.randint(0, len(self._starts), num_samples)
        if self.encoding_dims is None:
            return self._starts[indices], {}
        return self._starts[indices], {
            f'{k}_encoding': getattr(self, f'_{k}_encoding')[indices]
            for k in self.encoding_dims.keys()
        }

    def reencode_paths(
        self,
        history_encoders: Dict[str, HistoryEncoder],
    ):
        """Re-encode the replay buffer.

        Args:
            history_encoders: Name to history encoder.
        """
        self.encoding_dims = {}
        for path in self._paths:
            obs_seq = dm.torch_ify(path['observations'][np.newaxis])
            act_seq = dm.torch_ify(np.concatenate([
                np.zeros((1, 1, self._act_dim)),
                path['actions'][np.newaxis],
            ], axis=1))
            rew_seq = dm.torch_ify(np.concatenate([
                np.zeros((1, 1, 1)),
                path['rewards'][np.newaxis],
            ], axis=1))
            for k, encoder in history_encoders.items():
                with torch.no_grad():
                    encoding = encoder.forward(obs_seq, act_seq, rew_seq)[0]
                path[f'{k}_encoding'] = np.concatenate([
                    np.zeros((1, 1, encoding.shape[-1])),
                    dm.get_numpy(encoding),
                ], axis=1).squeeze(0)
                self.encoding_dims[k] = encoding.shape[-1]
        self.clear_buffer()
        for path in self._paths:
            self.add_paths(path)
