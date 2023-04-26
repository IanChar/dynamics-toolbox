"""
Measure correlation in errors.

Author: Ian Char
Date: April 25, 2023
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from dynamics_toolbox.env_wrappers.model_env import ModelEnv
from dynamics_toolbox.env_wrappers.wrapper_utils import get_terminal_from_env_name
from dynamics_toolbox.utils.sarsa_data_util import parse_into_trajectories
from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
from dynamics_toolbox.utils.storage.model_storage import (
    load_model_from_log_dir,
    load_ensemble_from_parent_dir,
)

###########################################################################
# %% Parse arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required='True')
parser.add_argument('--data_path', type=str, required='True')
parser.add_argument('--save_dir', type=str, default='err_corr')
parser.add_argument('--horizon', type=int, default=2)
parser.add_argument('--samples_per_start', type=int, default=10)
parser.add_argument('--miscal_fidelity', type=int, default=50)
parser.add_argument('--num_starts', type=int, default=100)
parser.add_argument('--is_ensemble', action='store_true')
parser.add_argument('--sampling_mode', type=str, default='sample_from_dist')
parser.add_argument('--ensemble_sampling_mode', type=str,
                    default='sample_member_every_trajectory')
parser.add_argument('--no_rewards', action='store_true')
parser.add_argument('--recal_constants', type=str, default=None)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

###########################################################################
# %% Load in model and data.
###########################################################################
print('Loading in model and data...')
paths = parse_into_trajectories(load_from_hdf5(args.data_path))
if args.is_ensemble:
    model = load_ensemble_from_parent_dir(
        parent_dir=args.model_path,
        sample_mode=args.ensemble_sampling_mode,
        member_sample_mode=args.sampling_mode,
    )
else:
    model = load_model_from_log_dir(path=args.model_path)
    model.sample_mode = args.sampling_mode
if args.recal_constants is not None:
    model.recal_constants = np.array([float(c)
                                      for c in args.recal_constants.split(',')])
model_env = ModelEnv(
    dynamics_model=model,
    penalty_coefficient=0.0,
    terminal_function=get_terminal_from_env_name(args.data_path),
    reward_is_first_dim=not args.no_rewards,
)

###########################################################################
# %% Prepare validation dataset.
###########################################################################
print('Preparing validation set...')
starts, offset_starts, obs, acts = [], [], [], []
paths = [path for path in paths if len(path['actions']) >= args.horizon + 1]
for _ in range(args.num_starts):
    rand_path = paths[np.random.randint(len(paths))]
    strt_idx = np.random.randint(len(rand_path['actions']) - args.horizon)
    starts.append(rand_path['observations'][strt_idx])
    offset_starts.append(rand_path['observations'][strt_idx+1])
    if args.no_rewards:
        obs.append(rand_path['next_observations'][strt_idx:strt_idx+args.horizon+1])
    else:
        obs.append(np.concatenate([
            rand_path['rewards'][strt_idx:strt_idx+args.horizon+1].reshape(-1, 1),
            rand_path['next_observations'][strt_idx:strt_idx+args.horizon+1],
        ], axis=-1))
    acts.append(rand_path['actions'][strt_idx:strt_idx+args.horizon+1])
starts, obs, acts, offset_starts = [np.array(ar) for ar in (starts, obs, acts,
                                                            offset_starts)]
starts = np.repeat(starts, args.samples_per_start, axis=0)
offset_starts = np.repeat(offset_starts, args.samples_per_start, axis=0)
acts = np.repeat(acts, args.samples_per_start, axis=0)

###########################################################################
# %% Generate the data.
###########################################################################
print('Generating data...')
rollouts = model_env.model_rollout_from_actions(
    num_rollouts=len(starts),
    actions=acts,
    starts=starts,
    show_progress=True,
)
if args.no_rewards:
    preds = rollouts['observations'][:, 1:].reshape(
        args.num_starts,
        args.samples_per_start,
        args.horizon+1,
        -1,
    )
else:
    preds = np.concatenate([
        rollouts['rewards'].reshape(args.num_starts, args.samples_per_start,
                                    args.horizon, 1),
        rollouts['observations'][:, 1:].reshape(
            args.num_starts,
            args.samples_per_start,
            args.horizon+1,
            -1,
        ),
    ], axis=-1)
offset_rollouts = model_env.model_rollout_from_actions(
    num_rollouts=len(offset_starts),
    actions=acts[:, 1:],
    starts=offset_starts,
    show_progress=True,
)
if args.no_rewards:
    offset_preds = offset_rollouts['observations'][:, 1:].reshape(
        args.num_starts,
        args.samples_per_start,
        args.horizon,
        -1,
    )
else:
    offset_preds = np.concatenate([
        offset_rollouts['rewards'].reshape(args.num_starts, args.samples_per_start,
                                           args.horizon, 1),
        offset_rollouts['observations'][:, 1:].reshape(
            args.num_starts,
            args.samples_per_start,
            args.horizon,
            -1,
        ),
    ], axis=-1)


###########################################################################
# %% Make scatter of first step errors x (shat2 - shat1) - (s2-s1)
###########################################################################
plt.style.use('seaborn')
obs = obs[:, np.newaxis]
first_errs = preds[:, :, 0] - obs[:, :, 0]
second_errs = (preds[:, :, 1] - preds[:, :, 0]) - (obs[:, :, 1] - obs[:, :, 0])
second_abs_errs = offset_preds[:, :, 0] - obs[:, :, 1]
os.makedirs(args.save_dir, exist_ok=True)
for dim in range(preds.shape[-1]):
    fig, axs = plt.subplots(1, 2)
    firsts = first_errs[..., dim].flatten()
    seconds = second_errs[..., dim].flatten()
    corr = np.corrcoef(firsts, seconds)[1, 0]
    m, c = np.linalg.lstsq(np.vstack([firsts, np.ones(len(firsts))]).T, seconds,
                           rcond=None)[0]
    ts = np.linspace(np.min(firsts), np.max(firsts), 100)
    axs[0].scatter(firsts, seconds, alpha=0.2)
    axs[0].plot(ts, m * ts + c, color='black', ls='--')
    axs[0].set_title(f'Rollout Error  Correlation={corr:0.2f}')
    axs[0].set_xlabel('First Timestep Error')
    axs[0].set_ylabel('Second Timestep Error')
    abs_seconds = second_abs_errs[..., dim].flatten()
    corr = np.corrcoef(firsts, abs_seconds)[1, 0]
    m, c = np.linalg.lstsq(np.vstack([firsts, np.ones(len(firsts))]).T, abs_seconds,
                           rcond=None)[0]
    axs[1].scatter(firsts, abs_seconds, alpha=0.2)
    axs[1].plot(ts, m * ts + c, color='black', ls='--')
    axs[1].set_title(f'Oracle Error  Correlation={corr:0.2f}')
    axs[1].set_xlabel('First Timestep Error')
    axs[1].set_ylabel('Second Timestep Error')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f'dim_{dim+1}.png'))
    plt.clf()
