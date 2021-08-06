"""
Construction functions for lightning.

Author: Ian Char
"""
from typing import Tuple
import os

from omegaconf import DictConfig, OmegaConf, open_dict
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from dynamics_toolbox.models import pl_models
from dynamics_toolbox.data import pl_data_modules
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel
from dynamics_toolbox.models.pl_models.pl_ensemble import FinitePlEnsemble
from dynamics_toolbox.utils.lightning.multi_early_stop import MultiMonitorEarlyStopping


def construct_all_pl_components_for_training(
        cfg: DictConfig
) -> Tuple[AbstractPlModel, LightningDataModule, pl.Trainer, LightningLoggerBase, DictConfig]:
    """Construct all components needed for training.

    Args:
        cfg: The configuration containing trainer, model, data_module info.

    Returns:
        * The data module.
        * The model to be used for training.
        * Trainer to be used for training.
        * The altered configuration.
    """
    data = getattr(pl_data_modules, cfg['data_module']['data_module_type'])(
            **cfg['data_module'])
    with open_dict(cfg):
        cfg['model']['input_dim'] = data.input_dim
        cfg['model']['output_dim'] = data.output_dim
    model = construct_pl_model(cfg['model'])
    callbacks = []
    if 'early_stopping' in cfg:
        callbacks.append(get_early_stopping_for_val_loss(cfg['early_stopping']))
    if cfg['logger'] == 'mlflow':
        from pytorch_lightning.loggers.mlflow import MLFlowLogger
        tracking_uri = cfg.get('tracking_uri', None)
        logger = MLFlowLogger(
            experiment_name=cfg['run_id'],
            tracking_uri=tracking_uri,
            save_dir=cfg['save_dir'],
        )
    else:
        logger = TensorBoardLogger(
                save_dir=cfg['save_dir'],
                name=cfg['run_id'],
        )
    trainer = pl.Trainer(**cfg['trainer'], logger=logger, callbacks=callbacks)
    return model, data, trainer, logger, cfg


def construct_pl_model(cfg: DictConfig) -> AbstractPlModel:
    """Construct a pytorch lightning model.

    Args:
        cfg: The configuration to create. Must have the following
            * model_type: the model type as registered in
              dynamics_toolbox/models/__init__.py
            * model_kwargs: The keyword arguments to pass to the model.
            * num_ensemble_members: The number of ensemble members to use.

    Returns:
        The pytorch lightning model.
    """
    if 'model_type' not in cfg:
        raise ValueError('Configuration does not have model_type')
    if 'num_ensemble_members' not in cfg:
        raise ValueError('Configuration does not have num_ensemble_members')
    if cfg['num_ensemble_members'] > 1:
        return FinitePlEnsemble(cfg)
    return getattr(pl_models, cfg['model_type'])(**cfg)


def get_early_stopping_for_val_loss(cfg: DictConfig) -> pl.callbacks.EarlyStopping:
    """Get an early stopping callback with a certain patience.

    Args:
        cfg: The configuration for early stopping.

    Returns:
        The early stopping callback to use in the trainer.
    """
    if cfg['num_ensemble_members'] > 1:
        return MultiMonitorEarlyStopping(
            monitors=[f'member{i}/val/loss' for i in range(cfg['num_ensemble_members'])],
            min_delta=cfg['min_delta'],
            patience=cfg['patience'],
            mode='min',
        )
    else:
        return pl.callbacks.EarlyStopping(
            monitor='val/loss',
            min_delta=cfg['min_delta'],
            patience=cfg['patience'],
            mode='min',
        )
