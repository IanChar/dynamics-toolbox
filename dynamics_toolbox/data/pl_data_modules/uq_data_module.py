"""
Data module for UQ datasets.

Author: Ian Char
Date: April 29, 2023
"""
from typing import Dict, Union, List, Sequence

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split

from dynamics_toolbox.utils.storage.qdata import load_from_hdf5


class UQDynamicsDataModule(LightningDataModule):

    def __init__(
            self,
            data_path: str,
            batch_size: int,
            val_proportion: float = 0.0,
            test_proportion: float = 0.0,
            num_workers: int = 1,
            pin_memory: bool = True,
            seed: int = 1,
            qset=None,
            **kwargs,
    ):
        """Constructor.

        Args:
            data_path: Path to the data.
            batch_size: Batch size.
            val_proportion: Proportion of data to use as validation.
            val_proportion: Proportion of data to use as test.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            seed: The seed.
            qset: The loaded in data.
            first_step_only: Whether we care about only the first step or about full
                sequences.
        """
        super().__init__()
        data = load_from_hdf5(data_path)
        self._obs, self._acts, self._means, self._stds, self._true = [
            data[k] for k in ('rollout_observations', 'actions', 'oracle_delta_means',
                              'oracle_delta_stds', 'true_deltas')
        ]
        self._obsacts = np.concatenate([self._obs, self._acts], axis=-1)
        self._val_proportion = val_proportion
        self._test_proportion = test_proportion
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._seed = seed
        data_size = len(self._xdata)
        self._num_val = int(data_size * val_proportion)
        self._num_te = int(data_size * test_proportion)
        self._num_tr = data_size - self._num_val - self._num_te
        self._tr_dataset, self._val_dataset, self._te_dataset = random_split(
            TensorDataset(
                torch.Tensor(self._obsacts),
                torch.Tensor(self._means),
                torch.Tensor(self._stds),
                torch.Tensor(self._true),
            ),
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
        return self._obs, self._acts, self._means, self._stds, self._true

    @property
    def input_data(self) -> np.array:
        """The input data.."""
        return self._obsacts

    @property
    def output_data(self) -> np.array:
        """The output data."""
        return self._stds

    @property
    def input_dim(self) -> int:
        """Observation dimension."""
        return self.input_data.shape[-1]

    @property
    def output_dim(self) -> int:
        return self._stds.shape[-1]

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
