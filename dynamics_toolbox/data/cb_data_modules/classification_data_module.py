"""
Data module for doing classification.

Author: Youngseog Chung
"""
from typing import Dict, Union, List, Sequence

import numpy as np
from imblearn.over_sampling import SMOTE
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
from catboost import Pool

from dynamics_toolbox.utils.storage.qdata import load_from_hdf5


class CBClassificationDataModule(LightningDataModule):

    def __init__(
            self,
            data_source: str,
            batch_size: int,
            val_proportion: float = 0.0,
            test_proportion: float = 0.0,
            num_workers: int = 1,
            pin_memory: bool = True,
            seed: int = 1,
            **kwargs,
    ):
        """Constructor.

        Args:
            data_source: Name of the data source as a path to the hdf5 file.
                The hdf5 file should contain:
                    * tr_x: The training x data.
                    * tr_y: The training y data.
                    * val_x: The validation x data.
                    * val_y: The validation y data.
                    * te_x (optional): The testing x data.
                    * te_y (optional): The testing y data.
            batch_size: Batch size.
            learn_rewards: Whether to include the rewards for learning in xdata.
            val_proportion: Proportion of data to use as validation.
            val_proportion: Proportion of data to use as test.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            seed: The seed.
        """
        super().__init__()
        dataset = load_from_hdf5(data_source)
        breakpoint()
        use_smote = bool(kwargs.get('smote', False))
        if use_smote:
            for split_key in ['tr', 'val', 'te']:
                curr_x = dataset[f'{split_key}_x']
                curr_y = (dataset[f'{split_key}_y'] >= 0.5).astype(int)  # TODO(ysc): I am just applying hard labels here, that's actually set in cfg['model'] for now
                x_re, y_re = SMOTE().fit_resample(curr_x, curr_y)
                dataset[f'{split_key}_x'] = x_re.reshape(-1, curr_x.shape[1])
                dataset[f'{split_key}_y'] = y_re.reshape(-1, curr_y.shape[1])
        self._xdata = dataset['tr_x']
        self._ydata = dataset['tr_y']
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
            TensorDataset(torch.Tensor(self._xdata), torch.Tensor(self._ydata)),
            [self._num_tr, self._num_val, self._num_te],
        )
        if 'val_x' in dataset and 'val_y' in dataset:
            self._val_dataset = TensorDataset(
                    torch.Tensor(dataset['val_x']),
                    torch.Tensor(dataset['val_y']),
            )
        if 'te_x' in dataset and 'te_y' in dataset:
            self._te_dataset = TensorDataset(
                    torch.Tensor(dataset['te_x']),
                    torch.Tensor(dataset['te_y']),
            )

        breakpoint()
        self.tr_x = self._tr_dataset.dataset.tensors[0].numpy()
        self.tr_y = self._tr_dataset.dataset.tensors[1].numpy()
        self.val_x = self._val_dataset.tensors[0].numpy()
        self.val_y = self._val_dataset.tensors[1].numpy()
        self.te_x = self._te_dataset.tensors[0].numpy()
        self.te_y = self._te_dataset.tensors[1].numpy()

        self.tr_pool = Pool(self.tr_x, self.tr_y)
        self.val_pool = Pool(self.val_x, self.val_y)
        self.te_pool = Pool(self.te_x, self.te_y)

        if 'num_classes' in kwargs:
            self._num_classes = int(kwargs.get('num_classes'))
        else:
            min_class = np.min(self._ydata)
            max_class = np.max(self._ydata)
            self._num_classes = int((max_class - min_class) + 1)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the training dataloader."""
        return DataLoader(
            self._tr_dataset,
            batch_size=self._batch_size,
            num_workers=self._num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=self._pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
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

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
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
        return self._xdata.shape[-1]

    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return self._num_classes

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
