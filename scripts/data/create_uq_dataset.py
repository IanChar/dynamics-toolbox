"""
Create a dataset that can be used for learning more sophisticated UQ.

Author: Ian Char
Date: April 29, 2023
"""
import argparse

import h5py
import numpy as np
from tqdm import tqdm

from dynamics_toolbox.env_wrappers.model_env import ModelEnv
from dynamics_toolbox.env_wrappers.wrapper_utils import get_terminal_from_env_name
from dynamics_toolbox.utils.sarsa_data_util import parse_into_snippet_datasets
from dynamics_toolbox.utils.storage.qdata import load_from_hdf5
from dynamics_toolbox.utils.storage.model_storage import (
    load_model_from_log_dir,
    load_ensemble_from_parent_dir,
)

###########################################################################
# %% Parse arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--horizon', type=int, default=10)
parser.add_argument('--is_ensemble', action='store_true')
parser.add_argument('--sampling_mode', type=str, default='sample_from_dist')
parser.add_argument('--ensemble_sampling_mode', type=str,
                    default='sample_member_every_step')
parser.add_argument('--no_rewards', action='store_true')
parser.add_argument('--recal_constants', type=str)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--unnormalize', action='store_true')
args = parser.parse_args()

###########################################################################
# %% Load in model and data.
###########################################################################
print('Loading in model and data...')
paths = parse_into_snippet_datasets(
    qset=load_from_hdf5(args.data_path),
    snippet_size=args.horizon,
    seed=args.seed,
    no_snippet_overlap=True,
)[0]
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
# %% Do rollouts.
###########################################################################
print('Rolling out with models...')
# Create the start states and trajectories.
obs = paths['observations']
starts = obs[:, 0]
nxts = paths['next_observations']
acts = paths['actions']
rews = paths['rewards'][..., np.newaxis]
rollouts = model_env.model_rollout_from_actions(
    num_rollouts=len(starts),
    actions=acts,
    starts=starts,
    show_progress=True,
)

###########################################################################
# %% Make an oracle rollout.
###########################################################################
oracle_means = []
oracle_stds = []
if args.unnormalize:
    norm_mean = getattr(model.normalizer, '1_offset').cpu().numpy()
    norm_std = getattr(model.normalizer, '1_scaling').cpu().numpy()
else:
    norm_mean = 0
    norm_std = 1
for h in tqdm(range(args.horizon), desc='Oracle Rollout...'):
    curr_ob, curr_act = obs[:, h], acts[:, h]
    _, info = model.predict(np.hstack([curr_ob, curr_act]))
    if len(info['mean_predictions'].shape) == 3:
        means, stds = info['mean_predictions'], info['std_predictions']
        means = means * norm_std + norm_mean
        stds *= norm_std
        members = len(means)
        mean_out = np.mean(means, axis=0)
        mean_var = np.mean(stds ** 2, axis=0)
        mean_sq = np.mean(means ** 2, axis=0) * (1 - 1 / members)
        mixing_term = 2 / (members ** 2) * np.sum(np.concatenate([np.array([
                means[i] * means[j]
                for j in range(i)])
            for i in range(1, members)]), axis=0)
        std_out = np.sqrt(mean_var + mean_sq - mixing_term)
    else:
        mean_out = info['mean_predictions'] * norm_std + norm_mean
        std_out = info['std_predictions'] * norm_std
    oracle_means.append(mean_out)
    oracle_stds.append(std_out)
oracle_means = np.array(oracle_means).transpose(1, 0, 2)
oracle_stds = np.array(oracle_stds).transpose(1, 0, 2)

###########################################################################
# %% Form the dataset.
###########################################################################
samples = rollouts['observations'][:, 1:] - rollouts['observations'][:, :-1]
if not args.no_rewards:
    samples = np.concatenate([rollouts['rewards'], samples], axis=-1)
if args.unnormalize:
    norm_mean = getattr(model.normalizer, '1_offset').cpu().numpy()
    norm_std = getattr(model.normalizer, '1_scaling').cpu().numpy()
else:
    norm_mean = 0
    norm_std = 1
if len(rollouts['info'][0]['mean_predictions'].shape) == 3:
    all_means = [inf['mean_predictions'] * norm_std + norm_mean
                 for inf in rollouts['info']]
    all_stds = [inf['std_predictions'] * norm_std
                for inf in rollouts['info']]
    # In this case, it is assumed we are dealing with a Gaussian mixture model.
    members, S, D = rollouts['info'][0]['mean_predictions'].shape
    mean_preds = np.array([np.mean(allm, axis=0)
                           for allm in all_means]).transpose(1, 0, 2)
    # For the standard prediction use the special case of law of total variance.
    std_preds = []
    for allm, alls in zip(all_means, all_stds):
        mean_var = np.mean(alls ** 2, axis=0)
        mean_sq = np.mean(allm ** 2, axis=0) * (1 - 1 / members)
        mixing_term = 2 / (members ** 2) * np.sum(np.concatenate([np.array([
                allm[i] * allm[j]
                for j in range(i)])
            for i in range(1, members)]), axis=0)
        std_preds.append(mean_var + mean_sq - mixing_term)
    std_preds = np.sqrt(np.array(std_preds).transpose(1, 0, 2))
else:
    mean_preds = np.array([inf['mean_predictions']
                           for inf in rollouts['info']]).transpose(1, 0, 2)
    mean_preds = mean_preds * norm_std + norm_mean
    std_preds = np.array([inf['std_predictions']
                          for inf in rollouts['info']]).transpose(1, 0, 2)
    std_preds *= norm_std
true_deltas = nxts - obs
if samples.shape[-1] > true_deltas.shape[-1]:
    true_deltas = np.concatenate([rews, true_deltas], axis=-1)
if not args.unnormalize:
    true_deltas /= getattr(model.normalizer, '1_scaling').cpu().numpy()
with h5py.File(args.save_path, 'w') as hdata:
    hdata.create_dataset('observations', data=obs)
    hdata.create_dataset('actions', data=acts)
    hdata.create_dataset('next_observations', data=nxts)
    hdata.create_dataset('rewards', data=rews)
    hdata.create_dataset('true_deltas', data=true_deltas)
    hdata.create_dataset('rollout_observations', data=rollouts['observations'])
    hdata.create_dataset('delta_samples', data=samples)
    hdata.create_dataset('delta_means', data=mean_preds)
    hdata.create_dataset('delta_stds', data=std_preds)
    hdata.create_dataset('oracle_delta_means', data=oracle_means)
    hdata.create_dataset('oracle_delta_stds', data=oracle_stds)
