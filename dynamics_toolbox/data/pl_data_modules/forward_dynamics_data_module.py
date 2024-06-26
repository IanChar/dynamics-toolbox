"""
Data module for training dynamics in forward sense (i.e. state, act -> nxt)

Author: Ian Char
"""
from typing import Dict, Union, List, Sequence

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split

from dynamics_toolbox.utils.storage.qdata import get_data_from_source


class ForwardDynamicsDataModule(LightningDataModule):

    def __init__(
            self,
            data_source: str,
            batch_size: int,
            learn_rewards: bool = False,
            val_proportion: float = 0.0,
            test_proportion: float = 0.0,
            num_workers: int = 1,
            pin_memory: bool = True,
            seed: int = 1,
            qset=None,
            account_for_d4rl_bug: bool = True,
            **kwargs,
    ):
        """Constructor.

        Args:
            data_source: Name of the data source, either as a string to a path or
                name of a d4rl env.
            batch_size: Batch size.
            learn_rewards: Whether to include the rewards for learning in xdata.
            val_proportion: Proportion of data to use as validation.
            val_proportion: Proportion of data to use as test.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            seed: The seed.
            qset: The loaded in data.
            account_for_d4rl_bug: It turns out d4rl does not log next_observations
                correctly if at a terminal state, so if you are using d4rl datasets
                (specifically walker or hopper) make sure this is True.
        """
        super().__init__()
        if qset is None:
            qset = get_data_from_source(data_source)
        if 'terminals' in qset and account_for_d4rl_bug:
            valid_idxs = np.argwhere(qset['terminals'] - 1).flatten()
            qset = {k: v[valid_idxs] for k, v in qset.items()}
        self._xdata = np.hstack([qset['observations'], qset['actions']])
        if learn_rewards:
            self._ydata = np.hstack([qset['rewards'].reshape(-1, 1),
                                     qset['next_observations'] - qset['observations']])
        else:
            self._ydata = qset['next_observations'] - qset['observations']
        self._learn_rewards = learn_rewards
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._seed = seed
        data_size = len(self._xdata)
        if val_proportion > 1:
            self._val_proportion = val_proportion / data_size
        else:
            self._val_proportion = val_proportion
        if test_proportion > 1:
            self._test_proportion = test_proportion / data_size
        else:
            self._test_proportion = test_proportion
        self._num_val = int(data_size * self._val_proportion)
        self._num_te = int(data_size * self._test_proportion)
        self._num_tr = data_size - self._num_val - self._num_te
        self._tr_dataset, self._val_dataset, self._te_dataset = random_split(
            TensorDataset(torch.Tensor(self._xdata), torch.Tensor(self._ydata)),
            [self._num_tr, self._num_val, self._num_te],
            generator=torch.Generator().manual_seed(seed),
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
    def val_data(self) -> Sequence[np.array]:
        """Get all of the data."""
        return self._val_dataset

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
        """Observation dimension."""
        return self._xdata.shape[-1]

    @property
    def output_dim(self) -> int:
        return self._ydata.shape[-1]

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
