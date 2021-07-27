"""
Some methods of training pytorch lightning models.

Author: Ian Char
"""
import argparse
import os
import pickle as pkl
from typing import NoReturn, Tuple

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.utilities.argparse import from_argparse_args

from dynamics_toolbox.data.pl_data_modules.forward_dynamics import ForwardDynamicsDataModule
from dynamics_toolbox.models.pl_models import PL_MODELS
from dynamics_toolbox.models.pl_models.pl_ensemble import FinitePlEnsemble
from dynamics_toolbox.utils.lightning.multi_early_stop import MultiMonitorEarlyStopping
from dynamics_toolbox.utils.storage.model_storage import save_config


def standard_model_train(args: argparse.Namespace) -> NoReturn:
    """Train a single model.
    Args:
         args: The output of parsing arguments.
    """
    model, data, trainer = assemble_pl_components(args)
    save_config(trainer, args)
    trainer.fit(model, data)


def assemble_pl_components(args: argparse.Namespace) -> Tuple[LightningModule, LightningDataModule, pl.Trainer]:
    """Assemble the components for training a PL model.
    Args:
        args: The output of parsing arguments.
    Returns:
        The model, data module, and the trainer.
    """
    data = ForwardDynamicsDataModule.from_argparse_args(args)
    args.input_dim = data.input_dim
    args.output_dim = data.output_dim
    model = assemble_model(args)
    if args.patience is not None:
        callbacks = [get_early_stopping_for_val_loss(args)]
    else:
        callbacks = None
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    return model, data, trainer


def assemble_model(args: argparse.Namespace) -> LightningModule:
    """
    Assemble the model to train.
    Args:
        args: The output of parsing arguments.
    Returns:
        The model.
    """
    if args.num_ensemble_members > 1:
        return FinitePlEnsemble(args)
    else:
        return from_argparse_args(PL_MODELS[args.model_type], args)


def get_early_stopping_for_val_loss(args: argparse.Namespace) -> pl.callbacks.EarlyStopping:
    """Get an early stopping callback with a certain patience.
    Args:
        args: The output of parsing arguments.
    Returns:
        The early stopping callback to use in the trainer.
    """
    if args.num_ensemble_members > 1:
        return MultiMonitorEarlyStopping(
            monitors=[f'member{i}/val/loss' for i in range(args.num_ensemble_members)],
            min_delta=args.min_delta,
            patience=args.patience,
            mode='min',
        )
    else:
        return pl.callbacks.EarlyStopping(
            monitor='val/loss',
            min_delta=args.min_delta,
            patience=args.patience,
            mode='min',
        )
