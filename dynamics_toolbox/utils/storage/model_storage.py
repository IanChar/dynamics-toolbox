"""
Utility for loading a model.

Author: Ian Char
"""
from argparse import Namespace
from copy import deepcopy
import os
import pickle as pkl
from typing import Optional

import numpy as np
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from dynamics_toolbox.models.abstract_model import\
        AbstractModel
from dynamics_toolbox.models import pl_models
from dynamics_toolbox.models.pl_models.simultaneous_ensemble import SimultaneousEnsemble


def load_model_from_log_dir(
    path: str,
    epoch: Optional[int] = None,
):
    """Load a model from a log directory.

    Args:
        path: The path to the log directory.
        epoch: Epoch of the checkpoint to load in. If not specified, load the
            last checkpoint recorded.

    Returns:
        The loaded dynamics model.
    """
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    path = os.path.join(path, 'checkpoints')
    checkpoints = os.listdir(path)
    epochs = [int(ck.split('-')[0].split('=')[1]) for ck in checkpoints]
    if epoch is not None:
        if epoch not in epochs:
            raise ValueError(f'Did not find epoch {epoch} in checkpoints.')
        epidx = epochs.index(epoch)
    else:
        epidx = np.argmax(epochs)
    path = os.path.join(path, checkpoints[epidx])
    return getattr(pl_models, cfg['model']['model_type']).load_from_checkpoint(
            path, **cfg['model'])


def load_model_from_tensorboard_log(
        run_id: str,
        version: Optional[int] = None,
        epoch: Optional[int] = None,
        default_root: str = 'trained_models',
) -> AbstractModel:
    """Load in a model.

    Args:
        run_id: The run_id of the model to load.
        path: The path to the model to load in.
        version: The version number to load in. If not specified, load the
            the latest version.
        epoch: Epoch of the checkpoint to load in. If not specified, load the
            last checkpoint recorded.
        default_root: The root directory to models.

    Returns:
        The loaded dynamics model.
    """
    path = os.path.join(default_root, run_id)
    path = os.path.join(path, 'lightning_logs')
    version_dirs = os.listdir(path)
    versions = [int(v.split('_')[1]) for v in version_dirs]
    if version is not None:
        if version not in versions:
            raise ValueError(f'Did not find version {version}')
        path = os.path.join(path, f'version_{version}')
    else:
        path = os.path.join(path, f'version_{max(versions)}')
    return load_model_from_log_dir(path, epoch=epoch)

