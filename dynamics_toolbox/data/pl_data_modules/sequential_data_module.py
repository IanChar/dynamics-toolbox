"""
Data module for training dynamics where multiple steps are given in a batch.

Author: Ian Char
"""
from typing import Dict, Union, List, Optional, Sequence

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from dynamics_toolbox.utils.sarsa_data_util import parse_into_snippet_datasets
from dynamics_toolbox.utils.storage.qdata import get_data_from_source


class SequentialDataModule(LightningDataModule):

    def __init__(
            self,
            data_source: str,
            batch_size: int,
            te_data_source: Optional[str] = None,
            learn_rewards: bool = False,
            snippet_size: Optional[int] = None,
            val_proportion: float = 0.0,
            test_proportion: float = 0.0,
            num_workers: int = 1,
            pin_memory: bool = True,
            seed: int = 1,
            allow_padding: bool = True,
            predict_deltas: bool = True,
            account_for_d4rl_bug: bool = True,
            **kwargs,
    ):
        """Constructor.

        Args:
            data_source: Name of the data source, either as a string to a path or
                name of a d4rl env.
            batch_size: Batch size.
            te_data_source: Path to data source of test set.
            learn_rewards: Whether to include the rewards for learning in xdata.
            snippet_size: How big each snippet from a trajectory should be. If it is
                None, set this to the smallest trajectory length.
            val_proportion: Proportion of data to use as validation.
            val_proportion: Proportion of data to use as test.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            seed: The seed.
            allow_padding: Whether to allow snippets to be incomplete and have padding.
            predict_deltas: Whether the labels should be deltas of the absolute
                next states.
        """
        super().__init__()
        qset = get_data_from_source(data_source)
        if 'terminals' in qset and account_for_d4rl_bug:
            valid_idxs = np.argwhere(qset['terminals'] - 1).flatten()
            qset = {k: v[valid_idxs] for k, v in qset.items()}
        self._snippets = parse_into_snippet_datasets(
            qset,
            snippet_size=snippet_size,
            val_proportion=val_proportion,
            test_proportion=test_proportion,
            allow_padding=allow_padding,
            seed=seed,
        )
        if te_data_source is not None:
            te_qset = get_data_from_source(te_data_source)
            self._snippets[-1] = parse_into_snippet_datasets(
                te_qset,
                snippet_size=snippet_size,
                val_proportion=0,
                test_proportion=0,
                allow_padding=allow_padding,
            )[0]
        datasets = []
        self._xdata, self._ydata = None, None
        for ds in self._snippets:
            if 'next_observations' in ds.keys():
                nexts = ds['next_observations']
            else:
                nexts = ds['next_obs']
            if predict_deltas:
                nexts -= ds['observations']
            if learn_rewards:
                if len(ds['rewards'].shape) < len(nexts.shape):
                    nexts = np.concatenate([ds['rewards'][..., np.newaxis], nexts], axis=-1)
                else:
                    nexts = np.concatenate([ds['rewards'], nexts], axis=-1)
            
            xdata = np.concatenate((ds['observations'], ds['actions']), axis=-1)
            datasets.append(TensorDataset(
                torch.Tensor(xdata),
                torch.Tensor(nexts),
                torch.Tensor(ds['mask']).unsqueeze(-1),
            ))
            if self._xdata is None:
                self._xdata = xdata.reshape(-1, xdata.shape[-1])
                self._ydata = nexts.reshape(-1, nexts.shape[-1])
        self._tr_dataset, self._val_dataset, self._te_dataset = datasets
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        self._learn_rewards = learn_rewards
        self._seed = seed

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
        return self._xdata.shape[-1]

    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return self._ydata.shape[-1]

    @property
    def num_train(self) -> int:
        """Number of training points."""
        return self._snippets[0]['observations'].shape[0]

    @property
    def num_validation(self) -> int:
        """Number of training points."""
        return self._snippets[1]['observations'].shape[0]

    @property
    def num_test(self) -> int:
        """Number of training points."""
        return self._snippets[2]['observations'].shape[0]
