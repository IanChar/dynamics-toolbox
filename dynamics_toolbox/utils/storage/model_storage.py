"""
Utility for loading a model.

Author: Ian Char
"""
import os
from typing import Optional, List

import hydra
from omegaconf import OmegaConf
import numpy as np

from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.abstract_model import\
        AbstractModel
from dynamics_toolbox.models.ensemble import Ensemble
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel


def load_model_from_log_dir(
    path: str,
    epoch: Optional[int] = None,
) -> AbstractPlModel:
    """Load a model from a log directory.

    Args:
        path: The path to the log directory.
        epoch: Epoch of the checkpoint to load in. If not specified, load the
            last checkpoint recorded. The checkpoint usually refers to the best epoch. 

    Returns:
        The loaded dynamics model.
    """
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    checkpoint_path = None
    for root, dirs, files in os.walk(path):
        if 'checkpoints' in dirs:
            checkpoint_path = os.path.join(root, 'checkpoints')
            break
    if checkpoint_path is None:
        raise ValueError(f'Checkpoint directory not found in {path}')
    checkpoints = os.listdir(checkpoint_path)
    if not len(checkpoints):
        raise ValueError(f'No checkpoints found in {checkpoint_path}')
    epochs = [int(ck.split('-')[0].split('=')[1]) for ck in checkpoints]
    if epoch is not None:
        if epoch not in epochs:
            raise ValueError(f'Did not find epoch {epoch} in checkpoints.')
        epidx = epochs.index(epoch)
    else:
        epidx = np.argmax(epochs)
    model_path = os.path.join(checkpoint_path, checkpoints[epidx])
    model = hydra.utils.instantiate(cfg['model'], _recursive_=False)
    return model.load_from_checkpoint(checkpoint_path=model_path, **cfg['model'])


def load_ensemble_from_list_of_log_dirs(
        paths: List[str],
        epochs: Optional[List[int]] = None,
        sample_mode: Optional[str] = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
        member_sample_mode: Optional[str] = None,
) -> Ensemble:
    """Load several models into an ensemble.

    Args:
        paths: List of paths to the log dirs.
        epochs: Epochs of the checkpoints to load in that correspond to the paths.
            Must be the same length as paths.
        sample_mode: The sampling mode for the ensemble.
    """
    paths.sort()
    epochs = [None for _ in paths] if epochs is None else epochs
    ensemble = Ensemble([load_model_from_log_dir(path, epoch)
                         for path, epoch in zip(paths, epochs)],
                        sample_mode=sample_mode)
    if member_sample_mode is not None:
        for memb in ensemble.members:
            memb.sample_mode = member_sample_mode
    return ensemble


def load_ensemble_from_parent_dir(
    parent_dir: str,
    sample_mode: Optional[str] = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
    member_sample_mode: Optional[str] = None,
) -> Ensemble:
    """Load all the models contained in the parent directory

    Args:
        parent_dir: The directory containing other directories of members.
        sample_mode: The sampling mode for the ensemble.
        member_sample_mode: How each of the children should sample.
    """
    children = os.listdir(parent_dir)
    paths = []
    for child in children:
        child = os.path.join(parent_dir, child)
        if os.path.isdir(child) and 'config.yaml' in os.listdir(child):
            paths.append(child)
    return load_ensemble_from_list_of_log_dirs(
        paths,
        sample_mode=sample_mode,
        member_sample_mode=member_sample_mode,
    )


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
