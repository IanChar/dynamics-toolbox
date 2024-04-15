"""
Data module for training autoregressive timeseries models
where multiple steps are given in a batch.

Author: Youngseog Chung
"""
from typing import Dict, Union, List, Optional, Sequence

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset

from dynamics_toolbox.utils.timeseries_data_util import (
    parse_into_timeseries_snippet_datasets,
)
from dynamics_toolbox.utils.storage.qdata import get_data_from_source


class TimeseriesDataModule(LightningDataModule):

    def __init__(
            self,
            data_source: str,
            batch_size: int,
            data_source_is_single_timeseries: bool,
            te_data_source: Optional[str] = None,
            te_data_source_is_single_timeseries: bool = False,
            snippet_size: Optional[int] = None,
            val_proportion: float = 0.0,
            test_proportion: float = 0.0,
            num_workers: int = 1,
            pin_memory: bool = True,
            seed: int = 1,
            allow_padding: bool = True,
            predict_deltas: bool = True,
            **kwargs,
    ):
        """Constructor.

        Args:
            data_source: Name of the data source, either as a string to a path or
                name of a d4rl env.
            batch_size: Batch size.
            data_source_is_single_timeseries: True means data_source consists of a
                single data-stream instead of e.g. concatenated trajectories.
            te_data_source: Path to data source of test set.
            te_data_source_is_single_timeseries: True means te_data_source consists of
                a single data-stream instead of e.g. concatenated trajectories.
            snippet_size: How big each snippet from a trajectory should be. If it is
                None, set this to the some default trajectory length.
                # TODO: decide which default length in L91 of timeseries_data_util.py
            val_proportion: Proportion of data to use as validation.
            test_proportion: Proportion of data to use as test.
            num_workers: Number of workers.
            pin_memory: Whether to pin memory.
            seed: The seed.
            allow_padding: Whether to allow snippets to be incomplete and have padding.
            predict_deltas: Whether the labels should be deltas of the absolute
                next states.
        """
        super().__init__()
        qset = get_data_from_source(data_source)
        self._snippets = parse_into_timeseries_snippet_datasets(
            qset,
            is_single_timeseries=data_source_is_single_timeseries,
            snippet_size=snippet_size,
            val_proportion=val_proportion,
            test_proportion=test_proportion,
            allow_padding=allow_padding,
        )
        if te_data_source is not None:
            te_qset = get_data_from_source(te_data_source)
            self._snippets[-1] = parse_into_timeseries_snippet_datasets(
                te_qset,
                is_single_timeseries=te_data_source_is_single_timeseries,
                snippet_size=snippet_size,
                val_proportion=0,
                test_proportion=0,
                allow_padding=allow_padding,
            )[0]
        datasets = []
        self._xdata, self._ydata = None, None
        for ds in self._snippets:
            nexts = ds['next_observations']
            if predict_deltas:
                nexts -= ds['observations']
            xdata = torch.Tensor(ds['observations'])
            nexts = torch.Tensor(nexts)
            masks = torch.Tensor(ds['mask']).unsqueeze(-1)
            if snippet_size == 1:  # In this case assume we are not working w seq model
                xdata.squeeze(1)
                nexts.squeeze(1)
                masks.squeeze(1)
            datasets.append(TensorDataset(xdata, nexts, masks))
            if self._xdata is None:
                self._xdata = xdata.numpy().reshape(-1, xdata.shape[-1])
                self._ydata = nexts.numpy().reshape(-1, nexts.shape[-1])
        self._tr_dataset, self._val_dataset, self._te_dataset = datasets
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._pin_memory = pin_memory
        # self._learn_rewards = learn_rewards
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
        """Get the validation dataloader."""
        if len(self._val_dataset):
            return DataLoader(
                self._val_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )

    def test_dataloader(
            self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        """Get the test dataloader."""
        if len(self._te_dataset):
            return DataLoader(
                self._te_dataset,
                batch_size=self._batch_size,
                num_workers=self._num_workers,
                shuffle=False,
                drop_last=False,
                pin_memory=self._pin_memory,
            )

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
