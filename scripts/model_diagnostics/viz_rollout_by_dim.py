"""
Rollout using a model and visualize each of the dimensions separately.

Author: Ian Char
Date: April 20, 2023
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
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
parser.add_argument('--save_dir', type=str, default='rollout_vizs')
parser.add_argument('--horizon', type=int, default=10)
parser.add_argument('--samples_per_start', type=int, default=15)
parser.add_argument('--num_starts', type=int, default=10)
parser.add_argument('--is_ensemble', action='store_true')
parser.add_argument('--sampling_mode', type=str, default='sample_from_dist')
parser.add_argument('--ensemble_sampling_mode', type=str,
                    default='sample_member_every_trajectory')
parser.add_argument('--no_rewards', action='store_true')
parser.add_argument('--plots_per_row', type=int, default=4)
parser.add_argument('--num_quantiles', type=int, default=10)
parser.add_argument('--show_samples', action='store_true')
parser.add_argument('--sample_alpha', type=float, default=0.2)
parser.add_argument('--show_dataset_max', action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()
np.random.seed(args.seed)

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
starts, obs, acts = [], [], []
paths = [path for path in paths if len(path['actions']) >= args.horizon]
for _ in range(args.num_starts):
    rand_path = paths[np.random.randint(len(paths))]
    strt_idx = np.random.randint(len(rand_path['actions']) - args.horizon)
    starts.append(rand_path['observations'][strt_idx])
    if args.no_rewards:
        obs.append(rand_path['next_observations'][strt_idx:strt_idx+args.horizon])
    else:
        obs.append(np.concatenate([
            rand_path['rewards'][strt_idx:strt_idx+args.horizon].reshape(-1, 1),
            rand_path['next_observations'][strt_idx:strt_idx+args.horizon],
        ], axis=-1))
    acts.append(rand_path['actions'][strt_idx:strt_idx+args.horizon])
starts, obs, acts = [np.array(ar) for ar in (starts, obs, acts)]
starts = np.repeat(starts, args.samples_per_start, axis=0)
acts = np.repeat(acts, args.samples_per_start, axis=0)

###########################################################################
# %% Generate the data.
###########################################################################
print('Generating data...')
breakpoint()
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
        args.horizon,
        -1,
    )
else:
    preds = np.concatenate([
        rollouts['rewards'].reshape(args.num_starts, args.samples_per_start,
                                    args.horizon, 1),
        rollouts['observations'][:, 1:].reshape(
            args.num_starts,
            args.samples_per_start,
            args.horizon,
            -1,
        ),
    ], axis=-1)

###########################################################################
# %% Plot each of the rollouts.
###########################################################################
plt.style.use('seaborn')
os.makedirs(args.save_dir, exist_ok=True)
num_dims = preds.shape[-1]
num_cols = args.plots_per_row
num_rows = int(np.ceil(num_dims / num_cols))
tsteps = np.arange(1, 1 + args.horizon)
quantiles = np.linspace(0.05, 0.95, args.num_quantiles)
cmap = pylab.cm.Blues(quantiles)
mins, maxs = (np.amin(obs.reshape(-1, obs.shape[-1]), axis=0),
              np.amax(obs.reshape(-1, obs.shape[-1]), axis=0))
for pnum in range(len(preds)):
    fig, axs = plt.subplots(num_rows, num_cols)
    for didx in range(num_dims):
        if num_rows == 1:
            ax = axs[didx]
        else:
            ax = axs[didx // num_cols, didx % num_cols]
        if args.show_samples:
            ax.plot(tsteps, preds[pnum, :, :, didx].T, color='royalblue',
                    alpha=args.sample_alpha)
        else:
            for quant, color in zip(quantiles, cmap):
                ax.fill_between(
                    tsteps,
                    np.quantile(preds[pnum, :, :, didx], 0.5 - quant / 2, axis=0),
                    np.quantile(preds[pnum, :, :, didx], 0.5 + quant / 2, axis=0),
                    color=color,
                    alpha=0.2,
                )
        if args.show_dataset_max:
            ax.axhline(mins[didx], ls=':', color='black')
            ax.axhline(maxs[didx], ls=':', color='black')
        ax.plot(tsteps, np.mean(preds[pnum, :, :, didx], axis=0), color='cyan',
                alpha=0.8)
        ax.plot(tsteps, obs[pnum, :, didx], ls='--', color='red')
        if not args.no_rewards:
            if didx == 0:
                ax.set_title('Rewards')
            else:
                ax.set_title(f'Dimension {didx}')
        else:
            ax.set_title(f'Dimension {didx + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, f'path_{pnum + 1}.png'))
    plt.clf()
