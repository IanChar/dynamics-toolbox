"""
A single progress bar that is kept throughout training.

Instead of creating a progress bar for each training and validation epoch. Have one
that only increases every every epoch.

Author: Ian Char
"""
from typing import List, Optional, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import tqdm


class SingleProgressBar(Callback):
    """A single progress bar that increases when an epoch is completed."""

    def __init__(
            self,
            num_epochs: int,
    ):
        """Constructor.

        Args:
            num_epochs: The number of training epochs there will be.
            track_statistics: List of statistics to display.
        """
        self._num_epochs = num_epochs
        self._pbar = None

    def setup(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            stage: Optional[str] = None,
    ) -> None:
        """Called when fit, validate test, predict or tune begins.

        Args:
            trainer: The pl trainer.
            pl_module: The pl module.
            stage: The stage.
        """
        self._pbar = tqdm.tqdm(total=self._num_epochs)

    def teardown(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            stage: Optional[str] = None,
    ) -> None:
        """Called when fit, validate test, predict or tune ends.

        Args:
            trainer: The pl trainer.
            pl_module: The pl module.
            stage: The stage.
        """
        if self._pbar is not None:
            self._pbar.close()

    def on_train_epoch_end(
            self,
            trainer: pl.Trainer,
            pl_module: pl.LightningModule,
            **kwargs
    ) -> None:
        """Called when the train epoch ends.

        Args:
            trainer: The pl trainer.
            pl_module: The pl module.
            **kwargs:
        """
        self._pbar.set_postfix(pl_module.get_progress_bar_dict())
        self._pbar.update(1)
