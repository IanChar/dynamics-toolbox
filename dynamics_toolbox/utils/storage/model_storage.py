"""
Utility for loading a model.

Author: Ian Char
"""
import os
import pickle as pkl
from typing import Dict, Optional, List

import hydra
from omegaconf import OmegaConf
import numpy as np

from dynamics_toolbox import DYNAMICS_TOOLBOX_PATH
from dynamics_toolbox.constants import sampling_modes
from dynamics_toolbox.models.abstract_model import\
        AbstractModel
from dynamics_toolbox.models.ensemble import Ensemble
from dynamics_toolbox.models.pl_models.abstract_pl_model import AbstractPlModel


def load_model_from_log_dir(
    path: str,
    epoch: Optional[int] = None,
    relative_path: bool = False,
    **kwargs
) -> AbstractPlModel:
    """Load a model from a log directory.

    Args:
        path: The path to the log directory.
        epoch: Epoch of the checkpoint to load in. If not specified, load the
            last checkpoint recorded.
        relative_path: Whether to look in the top level of the repo.

    Returns:
        The loaded dynamics model.
    """
    if relative_path:
        path = os.path.join(DYNAMICS_TOOLBOX_PATH, path)
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    checkpoint_paths = []
    for root, dirs, files in os.walk(path):
        if 'checkpoints' in dirs and len(os.listdir(os.path.join(root, 'checkpoints'))):
            checkpoint_paths.append(os.path.join(root, 'checkpoints'))
            break
    # Figure out each version of the paths.
    if len(checkpoint_paths) > 1:
        version_nums = []
        for cp in checkpoint_paths:
            version_num = -float('inf')
            for cp_dirs in cp.split('/'):
                if 'version_' in cp_dirs:
                    version_num = int(cp_dirs.split('_'))
                    break
            version_nums.append(version_num)
        checkpoint_path = checkpoint_paths[np.argmax(version_nums)]
    else:
        checkpoint_path = checkpoint_paths[0]
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
    should_recurse = cfg['model'].get('_recursive_', False)
    model = hydra.utils.instantiate(cfg['model'], _recursive_=should_recurse, **kwargs)
    if hasattr(model, 'wrapped_model'):
        wrapped_model = model.wrapped_model
    else:
        wrapped_model = None
    model = model.load_from_checkpoint(checkpoint_path=model_path, **cfg['model'])
    if wrapped_model is not None:
        model.set_wrapped_model(wrapped_model)
    return model


def load_ensemble_from_list_of_log_dirs(
    paths: List[str],
    epochs: Optional[List[int]] = None,
    sample_mode: Optional[str] = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
    member_sample_mode: Optional[str] = None,
    ignore_errors: bool = False,
) -> Ensemble:
    """Load several models into an ensemble.

    Args:
        paths: List of paths to the log dirs.
        epochs: Epochs of the checkpoints to load in that correspond to the paths.
            Must be the same length as paths.
        sample_mode: The sampling mode for the ensemble.
        ignore_errors: Ignore a member if we cannot load it in.
    """
    paths.sort()
    epochs = [None for _ in paths] if epochs is None else epochs
    members = []
    for path, epoch in zip(paths, epochs):
        try:
            members.append(load_model_from_log_dir(path, epoch))
        except BaseException as exc:
            if ignore_errors:
                continue
            raise exc
    ensemble = Ensemble(members, sample_mode=sample_mode)
    if member_sample_mode is not None:
        for memb in ensemble.members:
            memb.sample_mode = member_sample_mode
    return ensemble


def load_ensemble_from_parent_dir(
    parent_dir: str,
    sample_mode: Optional[str] = sampling_modes.SAMPLE_MEMBER_EVERY_TRAJECTORY,
    member_sample_mode: Optional[str] = None,
    load_n_best_models: Optional[int] = None,
    select_statistic: str = 'val/nll',
    lower_stat_is_better: bool = True,
    relative_path: bool = True,
    ignore_errors: bool = False,
) -> Ensemble:
    """Load all the models contained in the parent directory

    Args:
        parent_dir: The directory containing other directories of members.
        sample_mode: The sampling mode for the ensemble.
        member_sample_mode: How each of the children should sample.
        load_n_best_models: Load only the N best models based on validation score.
            Load all models if this is not specified.
        select_statistic: Statistic to evaluate based on.
        lower_stat_is_better: Whether the lower the statistic the better.
        relative_path: Whether to look in the top level of the repo.
    """
    if relative_path:
        parent_dir = os.path.join(DYNAMICS_TOOLBOX_PATH, parent_dir)
    children = os.listdir(parent_dir)
    paths = []
    if load_n_best_models:
        children_stats = [
            get_model_stats(os.path.join(parent_dir, child))[select_statistic]
            for child in children
        ]
        best_idxs = np.argsort(children_stats)
        if not lower_stat_is_better:
            best_idxs = best_idxs[::-1]
        children = [children[cidx] for cidx in best_idxs[:load_n_best_models]]
    for child in children:
        child = os.path.join(parent_dir, child)
        if os.path.isdir(child) and 'config.yaml' in os.listdir(child):
            paths.append(child)
    return load_ensemble_from_list_of_log_dirs(
        paths,
        sample_mode=sample_mode,
        member_sample_mode=member_sample_mode,
        ignore_errors=ignore_errors,
    )


def load_model_from_tensorboard_log(
        run_id: str,
        version: Optional[int] = None,
        epoch: Optional[int] = None,
        default_root: str = 'trained_models',
        relative_path: bool = True,
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
        relative_path: Whether to look in the top level of the repo.

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


def get_model_stats(
    model_dir: str,
    stat_type: str = 'val',
) -> Dict[str, float]:
    """Load in final statistics about the model.

    Args:
        model_dir: Path pointing to the model.
        stat_type: The type of statistics, either val or te.

    Returns: Dictionary of statitics.
    """
    stats = None
    stat_name = f'{stat_type}_eval_stats.pkl'
    for root, dirs, files in os.walk(model_dir):
        if stat_name in files:
            with open(os.path.join(root, stat_name), 'rb') as f:
                stats = pkl.load(f)
            break
    if isinstance(stats, list):
        stats = stats[0]
    return stats
