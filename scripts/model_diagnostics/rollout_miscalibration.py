"""
Evaluate miscalibration on the rollouts of models wrt to dynamics.

Author: Ian Char
Date: April 17, 2023
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dynamics_toolbox.env_wrappers.model_env import ModelEnv
from dynamics_toolbox.env_wrappers.wrapper_utils import get_terminal_from_env_name
from dynamics_toolbox.metrics.uq_metrics import miscalibration_from_samples
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
parser.add_argument('--save_dir', type=str, default='miscal_figures')
parser.add_argument('--horizon', type=int, default=10)
parser.add_argument('--samples_per_start', type=int, default=100)
parser.add_argument('--miscal_fidelity', type=int, default=50)
parser.add_argument('--num_starts', type=int, default=1000)
parser.add_argument('--is_ensemble', action='store_true')
parser.add_argument('--sampling_mode', type=str, default='sample_from_dist')
parser.add_argument('--ensemble_sampling_mode', type=str,
                    default='sample_member_every_trajectory')
parser.add_argument('--no_rewards', action='store_true')
parser.add_argument('--recal_constants', type=str, default=None)
parser.add_argument('--wrapper_path', type=str, default=None)
parser.add_argument('--no_recal', action='store_true')
parser.add_argument('--no_corr', action='store_true')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

###########################################################################
# %% Load in model and data.
###########################################################################
np.random.seed(args.seed)
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
if args.wrapper_path is not None:
    wrapper = load_model_from_log_dir(path=args.wrapper_path)
    wrapper.set_wrapped_model(model)
    wrapper.apply_corr = not args.no_corr
    wrapper.apply_recal = not args.no_recal
    model = wrapper
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
# %% Calculate miscalibration.
###########################################################################
print('Calculating scores...')
miscals, overconfs, avg_ev, median_ev, best_ev = [[] for _ in range(5)]
for h in tqdm(range(args.horizon)):
    miscal, overconf = miscalibration_from_samples(
        preds[:, :, h],
        obs[:, h],
        include_overconfidence_scores=True,
        fidelity=args.miscal_fidelity,
        use_intervals=True,
    )
    miscals.append(miscal)
    overconfs.append(overconf)
    all_resids = np.array([
        preds[:, :, h, d] - obs[:, h, d][:, np.newaxis]
        for d in range(preds.shape[-1])
    ])
    avg_resids = np.mean(all_resids, axis=-1)
    mode_resids = np.median(all_resids, axis=-1)
    min_mag_idxs = np.argmin(np.abs(all_resids), axis=-1)
    best_resids = np.take_along_axis(
        all_resids,
        np.expand_dims(min_mag_idxs, axis=-1),
        axis=-1,
    )
    avg_ev.append(np.array([
        (np.var(avg_resids[d] - np.mean(avg_resids[d]))
         / np.var(obs[:, h, d] - np.mean(obs[:, h, d])))
        for d in range(preds.shape[-1])
    ]))
    median_ev.append(np.array([
        (np.var(mode_resids[d] - np.mean(mode_resids[d]))
         / np.var(obs[:, h, d] - np.mean(obs[:, h, d])))
        for d in range(preds.shape[-1])
    ]))
    best_ev.append(np.array([
        (np.var(best_resids[d] - np.mean(best_resids[d]))
         / np.var(obs[:, h, d] - np.mean(obs[:, h, d])))
        for d in range(preds.shape[-1])
    ]))
miscals, overconfs, avg_ev, median_ev, best_ev = [
    np.array(ar) for ar in (miscals, overconfs, avg_ev, median_ev, best_ev)]

###########################################################################
# %% Make miscalibration plots.
###########################################################################
plt.style.use('seaborn')


def plot_miscal(mcal, oconf, aev, mev, bev, title, save_path=None, show=False):
    tstep = np.arange(1, len(mcal) + 1)
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(tstep, mcal)
    axs[0].plot(tstep, oconf, ls='--', alpha=0.6)
    axs[0].fill_between(tstep, np.zeros(len(oconf)), oconf, color='red', alpha=0.4)
    axs[0].fill_between(tstep, oconf, mcal, color='blue', alpha=0.4)
    axs[0].set_xlabel('Time Step')
    axs[0].set_ylabel('Miscalibration')
    axs[0].set_title(f'{title} Total Area={np.sum(mcal):0.2f}'
                     f' Overconfidence={np.sum(oconf):0.2f}')
    axs[0].set_ylim([0, 0.5])
    axs[1].plot(tstep, aev, label='Average')
    axs[1].plot(tstep, mev, label='Median')
    axs[1].plot(tstep, bev, label='Best')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('1 - EV')
    axs[1].legend()
    axs[1].set_ylim([-0.1, 1.5])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.clf()


os.makedirs(args.save_dir, exist_ok=True)
for dim in range(miscals.shape[1]):
    plot_miscal(miscals[:, dim], overconfs[:, dim], avg_ev[:, dim],
                median_ev[:, dim], best_ev[:, dim],
                title=f'Dimension {dim + 1}',
                save_path=os.path.join(args.save_dir, f'dim_{dim+1}.png'))
plot_miscal(np.mean(miscals, axis=-1), np.mean(overconfs, axis=-1),
            np.mean(avg_ev, axis=-1),
            np.mean(median_ev, axis=-1),
            np.mean(best_ev, axis=-1),
            title='Average',
            save_path=os.path.join(args.save_dir, 'average.png'))
