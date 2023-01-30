"""
Construction functions for lightning.

Author: Ian Char
"""
from typing import Tuple
import os

import hydra.utils
import numpy as np
from omegaconf import DictConfig, open_dict
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch

from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
from dynamics_toolbox.utils.lightning.single_progress_bar import SingleProgressBar
from dynamics_toolbox.utils.pytorch.modules.normalizer import (
    Normalizer, InputNormalizer
)


def construct_all_pl_components_for_training(
        cfg: DictConfig
) -> Tuple[AbstractPlModel, LightningDataModule, pl.Trainer, LightningLoggerBase,
           DictConfig]:
    """Construct all components needed for training.

    Args:
        cfg: The configuration containing trainer, model, data_module info.

    Returns:
        * The data module.
        * The model to be used for training.
        * Trainer to be used for training.
        * The altered configuration.
    """
    data_module = hydra.utils.instantiate(cfg['data_module'])
    with open_dict(cfg):
        cfg['model']['input_dim'] = data_module.input_dim
        cfg['model']['output_dim'] = data_module.output_dim
        if 'member_config' in cfg['model']:
            cfg['model']['member_config']['input_dim'] = data_module.input_dim
            cfg['model']['member_config']['output_dim'] = data_module.output_dim
    if 'normalization' not in cfg:
        normalizer = None
    elif cfg['normalization'] == 'standardize':
        normalizer = Normalizer(
            [(torch.Tensor(np.mean(b.reshape(-1, b.shape[-1]), axis=0)),
              torch.Tensor(np.std(b.reshape(-1, b.shape[-1]), axis=0)))
             for b in data_module.data]
        )
    elif cfg['normalization'] == 'standardize_input':
        b = data_module.data[0]
        normalizer = InputNormalizer(
            [(torch.Tensor(np.mean(b.reshape(-1, b.shape[-1]), axis=0)),
             torch.Tensor(np.std(b.reshape(-1, b.shape[-1]), axis=0)))
             ]
        )
    else:
        raise ValueError(f'Normalization scheme {cfg["normalization"]} not found.')
    model = hydra.utils.instantiate(cfg['model'], normalizer=normalizer,
                                    _recursive_=False)
    callbacks = []
    if 'early_stopping' in cfg:
        callbacks.append(get_early_stopping_for_val_loss(cfg['early_stopping']))
    max_epochs = (1000 if 'max_epochs' not in cfg['trainer']
                  else cfg['trainer']['max_epochs'])
    if hasattr(model, 'set_additional_model_params'):
        model.set_additional_model_params({'iterations': max_epochs})
    if data_module.num_validation > 0:
        callbacks.append(ModelCheckpoint(monitor='val/loss'))
    callbacks.append(SingleProgressBar(max_epochs))
    if cfg['logger'] == 'mlflow':
        from pytorch_lightning.loggers.mlflow import MLFlowLogger
        experiment = cfg.get('experiment_name', 'experiment')
        if 'run_name' in cfg:
            run_name = cfg['run_name']
        else:
            run_name = cfg.get('name', 'temp')
        logger = MLFlowLogger(
            experiment_name=experiment,
            tracking_uri=cfg.get('tracking_uri', None),
            save_dir=cfg['save_dir'],
            run_name=run_name,
        )
    else:
        name = cfg['name']
        logger = TensorBoardLogger(save_dir=cfg['save_dir'], name=name)
    trainer = pl.Trainer(
        **cfg['trainer'],
        logger=logger,
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
    )
    return model, data_module, trainer, logger, cfg


def get_early_stopping_for_val_loss(cfg: DictConfig) -> pl.callbacks.EarlyStopping:
    """Get an early stopping callback with a certain patience.

    Args:
        cfg: The configuration for early stopping.

    Returns:
        The early stopping callback to use in the trainer.
    """
    monitor = cfg.get('monitor', 'val/loss')
    mode = cfg.get('mode', 'min')
    return pl.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=cfg['min_delta'],
        patience=cfg['patience'],
        mode=mode,
    )
