"""
Collect data from a saved policy.

Author: Ian Char
Date: May 6, 2023
"""
import argparse
from collections import defaultdict
import os

import gym
import h5py
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm

from dynamics_toolbox.rl.util.misc import load_policy
from dynamics_toolbox.rl.util.gym_util import gym_rollout_from_policy
import dynamics_toolbox.rl.envs


###########################################################################
# %% Load in arguments.
###########################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--run_dir', type=str, required=True)
parser.add_argument('--save_path', type=str, required=True)
parser.add_argument('--num_paths', type=int, required=True)
parser.add_argument('--path_length', type=int)
parser.add_argument('--checkpoint', type=int)
parser.add_argument('--stochastic', action='store_true')
args = parser.parse_args()

###########################################################################
# %% Load in the policy and make the environment.
###########################################################################
policy = load_policy(args.run_dir, args.checkpoint)
policy.deterministic = not args.stochastic
cfg = OmegaConf.load(os.path.join(args.run_dir, 'config.yaml'))
env = gym.make(cfg['env_name'])
path_length = args.path_length if args.path_length else cfg['train_loop']['horizon']

###########################################################################
# %% Collect paths.
###########################################################################
paths = defaultdict(list)
for path_num in tqdm(range(args.num_paths), desc='Collecting Paths'):
    path = gym_rollout_from_policy(env, policy, path_length)
    for k, v in path.items():
        if k == 'observations':
            paths['observations'].append(v[:-1])
            paths['next_observations'].append(v[1:])
        else:
            paths[k].append(v)

###########################################################################
# %% Save paths.
###########################################################################
with h5py.File(args.save_path, 'w') as hdata:
    for k, v in paths.items():
        hdata.create_dataset(k, data=np.vstack(v))
