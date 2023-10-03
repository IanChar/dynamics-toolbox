"""
Script for collecting offline data. Adapted from rlkit's run_policy script.

Author: Ian Char
Date: 9/10/2020
"""
import argparse

import gym
import h5py
import numpy as np
import torch
from tqdm import tqdm

import dynamics_toolbox.rl.envs
from dynamics_toolbox.rl.modules.policies.random_policy import RandomPolicy


def load_in_policy(args):
    if args.is_rlkit_policy:
        from rlkit.torch.pytorch_util import set_gpu_mode
        data = torch.load(args.policy_path, map_location='cpu')
        policy = data['evaluation/policy']
        if args.gpu:
            set_gpu_mode(True)
            policy.cuda()
    else:
        raise NotImplementedError('TODO')
    return policy


def collect_data(args):
    np.random.seed(args.seed)
    env = gym.make(args.env)
    from dynamics_toolbox.rl.modules.policies.pid_policy import PID
    # policy = PID(0.1, 0.001, 0.001, ob_idx=0, target_dist=(1.5, 2.1))  # ID
    # policy = PID(0.2, 0.0, 0.25, ob_idx=0, target_dist=(1.8, 2.15))  # OOD
    if args.policy_path is not None:
        policy = load_in_policy(args)
    else:
        policy = RandomPolicy(action_space=env.action_space)
    print("Policy loaded")
    observations, actions, rewards, next_observations, terminals, timeouts =\
        [[] for _ in range(6)]
    pbar = tqdm(total=args.num_collects)
    earlyterms = []
    neps = 0
    returns = []
    while len(observations) < args.num_collects:
        done = False
        s, _ = env.reset()
        policy.reset()
        neps += 1
        t = 0
        timeout = False
        ret = 0
        while not done and t < args.path_length and not timeout:
            a, _ = policy.get_action(s)
            n, r, done, truncated = env.step(a)[:4]
            ret += r
            t += 1
            ret += r
            if (t + 1 >= args.path_length
                    or len(observations) + 1 >= args.num_collects):
                timeout = True
                done = args.terminate_at_horizon
            else:
                if done:
                    earlyterms.append(t)
                timeout = False
            observations.append(s)
            actions.append(a)
            rewards.append(r)
            next_observations.append(n)
            terminals.append(done)
            timeouts.append(timeout)
            s = n
            pbar.update(1)
            if len(observations) == args.num_collects:
                break
        returns.append(ret)
    print(f"{neps} episodes")
    print('Returns: %f +- %f' % (np.mean(returns), np.std(returns)))
    with h5py.File(args.save_path, 'w') as wd:
        wd.create_dataset('observations', data=np.vstack(observations))
        wd.create_dataset('actions', data=np.vstack(actions))
        wd.create_dataset('rewards', data=np.vstack(rewards))
        wd.create_dataset('next_observations', data=np.vstack(next_observations))
        wd.create_dataset('terminals', data=np.vstack(terminals))
        wd.create_dataset('timeouts', data=np.vstack(timeouts))
    print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--env', type=str)
    parser.add_argument('--policy_path', type=str, default=None,
                        help='If not specified use a random policy.')
    parser.add_argument('--path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--num_collects', type=int)
    parser.add_argument('--terminate_at_horizon', action='store_true',
                        help='States at end of time limit marked as terminal.')
    parser.add_argument('--is_rlkit_policy', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    collect_data(args)
