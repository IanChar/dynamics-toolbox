"""
Trainer for online model based RL.

Author: Ian Char
Date: April 23, 2023
"""
from typing import Dict, Optional

import torch
from pytorch_lightning import LightningDataModule

from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm


class DynamicsTrainer:

    def __init__(
        self,
        patience: int,
        max_epochs: Optional[int] = None,
        val_monitor: str = 'val/loss',
        new_optimizer_each_train: bool = False,
    ):
        """Constructor.

        Args:
            patience: How long to wait on the val loss before overfitting.
            max_epochs: Maximum amount of epochs to train for.
            val_monitor: The validation statistic to look at for overfitting.
            new_optimizer_each_train: Whether a new optimizer should be initialized
                for every train.
        """
        self._patience = patience
        self._max_epochs = max_epochs
        self._val_monitor = val_monitor
        self._new_optimizer_each_train = new_optimizer_each_train
        self._optimizer = None

    def fit(
        self,
        model: AbstractPlModel,
        data_module: LightningDataModule,
    ) -> Dict[str, float]:
        """Fit the model to the data.

        Args:
            model: The model to train.
            data_module: The data module to use to train the model.

        Returns: Dictionary of statisics about the training.
        """
        if self._optimizer is None or self._new_optimizer_each_train:
            self._optimizer = model.configure_optimizers()
        epoch_num = 0
        running = True
        best_loss, best_epoch = float('inf'), None
        tr_loss, val_loss = None, None
        while running:
            epoch_num += 1
            # Training
            model.train()
            tr_loss = 0
            tr_loader = data_module.train_dataloader()
            for batch in tr_loader:
                pt_batch = [b.to(dm.device) for b in batch]
                net_out = model.get_net_out(pt_batch)
                loss, _ = model.loss(net_out, pt_batch)
                tr_loss += loss.item()
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            tr_loss /= len(tr_loader)
            # Validation batch.
            model.eval()
            val_loss = 0
            val_loader = data_module.val_dataloader()
            for batch in val_loader:
                pt_batch = [b.to(dm.device) for b in batch]
                with torch.no_grad():
                    net_out = model.get_net_out(pt_batch)
                    loss, _ = model.loss(net_out, pt_batch)
                val_loss += loss.item()
            val_loss /= len(val_loader)
            # See if that was the best so far.
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch_num
            running = (best_epoch - epoch_num > self._patience
                       or (self._max_epochs is not None
                           and (epoch_num >= self._max_epochs)))
        return {'Model/train_loss': tr_loss, 'Model/val_loss': val_loss}
