"""
Data module for training dynamics where multiple steps are given in a batch.

Author: Ian Char
"""
from typing import Dict, Union, List, Optional, Sequence

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split

from dynamics_toolbox.utils.gym_util import parse_into_trajectories
from dynamics_toolbox.utils.storage.qdata import get_data_from_source


class SequentialDataModule(LightningDataModule):

    def __init__(
            self,
            data_source: str,
            batch_size: int,
            learn_rewards: bool = False,
            snippet_size: Optional[int] = None,
            val_proportion: float = 0.0,
            test_proportion: float = 0.0,
            num_workers: int = 1,
            pin_memory: bool = True,
            seed: int = 1,
            **kwargs,
    ):
        """Constructor.

        Args:
            data_source: Name of the data source, either as a string to a path or
                name of a d4rl env.
            batch_size: Batch size.
            learn_rewards: Whether to include the rewards for learning in xdata.
            snippet_size: How big each snippet from a trajectory should be. If it is
                None, set this to the smallest trajectory length.
            val_proportion: Proportion of data to use as validation.
            val_proportion: Proportion of data to use as test.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            seed: The seed.
        """
        super().__init__()
        qset = get_data_from_source(data_source)
        trajectories = parse_into_trajectories(qset)
        traj_lengths = [len(traj['observations']) for traj in trajectories]
        if snippet_size is None:
            snippet_size = np.min(traj_lengths)
        observations = []
        actions = []
        rewards = []
        terminals = []
        for traj, traj_len in zip(trajectories, traj_lengths):
            for idx in range(traj_len + 1 - snippet_size):
                observations.append(np.vstack([
                    traj['observations'][[idx]],
                    traj['next_observations'][idx:idx + snippet_size]
                ]))
                actions.append(traj['actions'][idx:idx + snippet_size])
                rewards.append(traj['rewards'][idx:idx + snippet_size])
                terminals.append(traj['terminals'][idx:idx + snippet_size])
        self._observations = np.array(observations)
        self._actions = np.array(actions)
        self._rewards = np.array(rewards)
        self._terminals = np.array(terminals)
        self._xdata = np.concatenate([self._observations[:, :-1], self._actions],
                                     axis=-1)
        self._ydata = self._observations[:, 1:] - self._observations[:, :-1]
        if learn_rewards:
            self._ydata = np.concatenate([self._rewards, self._ydata], axis=-1)
        self._val_proportion = val_proportion
        self._test_proportion = test_proportion
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._seed = seed
        data_size = len(self._observations)
        self._num_val = int(data_size * val_proportion)
        self._num_te = int(data_size * test_proportion)
        self._num_tr = data_size - self._num_val - self._num_te
        self._tr_dataset, self._val_dataset, self._te_dataset = random_split(
            TensorDataset(torch.Tensor(self._xdata), torch.Tensor(self._ydata)),
            [self._num_tr, self._num_val, self._num_te],
        )

    def train_dataloader(
            self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        return DataLoader(
            self._tr_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        if len(self._val_dataset):
            return DataLoader(
                self._val_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )
        else:
            None

    def test_dataloader(
            self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        if len(self._te_dataset):
            return DataLoader(
                self._te_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )
        else:
            None

    @property
    def data(self) -> Sequence[np.array]:
        """Get all of the data."""
        return self._xdata, self._ydata

    @property
    def input_data(self) -> np.array:
        """The input data.."""
        return self._xdata

    @property
    def output_data(self) -> np.array:
        """The output data."""
        return self._ydata

    @property
    def input_dim(self) -> int:
        """Input dimension."""
        return self._observations.shape[-1] + self._actions.shape[-1]

    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return self._observations.shape[-1]

    @property
    def num_train(self) -> int:
        """Number of training points."""
        return self._num_tr

    @property
    def num_validation(self) -> int:
        """Number of training points."""
        return self._num_val

    @property
    def num_test(self) -> int:
        """Number of training points."""
        return self._num_te
