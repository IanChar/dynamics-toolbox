"""
Utilities for interacting with gym environments.

Author: Ian Char
Date: April 6, 2023
"""
import gym
from typing import Dict, List, Optional, Tuple

import numpy as np

from dynamics_toolbox.rl.modules.policies.abstract_policy import Policy


def gym_rollout_from_policy(
    env: gym.Env,
    policy: Policy,
    horizon: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Do one rollout in a gym environment.

    Args:
        env: The gym environment.
        policy: The policy to use.
        horizon: Horizon of the episode. If None, will run until terminal.

    Returns:
        * obs: Observation ndarray of shape (horizon + 1, obs_dim)
        * act: Action ndarray of shape (horizon, obs_dim)
        * rew: Reward ndarray of shape (horizon, 1)
        * terminal: Terminal ndarray of shape (horizon, 1)
        * logprobs: Log probabilities of taking actions w shape (horizon, 1)
    """
    obs, acts, rews, terms, logprobs = [[] for _ in range(5)]
    start_obs, _ = env.reset()
    obs.append(start_obs)
    policy.eval()
    policy.reset()
    running = True
    h = 0
    while running:
        act, logprob = policy.get_actions(obs[-1])
        acts.append(act)
        logprobs.append(logprob)
        ob, rew, term, _, _ = env.step(unnormalize_action(env, act))
        obs.append(ob)
        rews.append(rew)
        terms.append(term)
        h += 1
        running = (not term) and ((horizon is None) or h < horizon)
        policy.get_reward_feedback(rew)
    return {
        'observations': np.array(obs),
        'actions': np.array(acts),
        'rewards': np.array(rews).reshape(-1, 1),
        'terminals': np.array(terms).reshape(-1, 1),
        'logprobs': np.array(logprobs).reshape(-1, 1),
    }


def explore_gym_until_threshold_met(
    env: gym.Env,
    policy: Policy,
    num_steps: int,
    horizon: Optional[int] = None,
) -> List[Dict[str, np.ndarray]]:
    """Do exploration until we have met num_steps threshold.

    Args:
        env: The gym environment.
        policy: The policy to use.
        num_steps: Number of steps to achieve.
        horizon: Horizon of the episode. If None, will run until terminal.

    Returns: List of path information each with
        * obs: Observation ndarray of shape (horizon + 1, obs_dim)
        * act: Action ndarray of shape (horizon, obs_dim)
        * rew: Reward ndarray of shape (horizon, 1)
        * terminal: Terminal ndarray of shape (horizon, 1)
        * logprobs: Log probabilities of taking actions w shape (horizon, 1)
    """
    num_steps_taken = 0
    paths = []
    horizon = float('inf') if horizon is None else horizon
    policy.deterministic = False
    while num_steps_taken < num_steps:
        steps_this_ep = min(num_steps - num_steps_taken, horizon)
        paths.append(gym_rollout_from_policy(
            env,
            policy,
            steps_this_ep
        ))
        num_steps_taken += len(paths[-1]['rewards'])
    return paths


def evaluate_policy_in_gym(
    env: gym.Env,
    policy: Policy,
    num_eps: int,
    horizon: Optional[int] = None,
    env_eval: bool = True,
) -> Tuple[float, float]:
    """Do one rollout in a gym environment.

    Args:
        env: The gym environment.
        policy: The policy to use.
        num_eps: Number of episodes to evaluate on.
        horizon: Horizon of the episode. If None, will run until terminal.
        env_eval: If the environment has an evaluation mode, whether it should
            be activated.

    Returns: Average and standard deviation of score across episodes.
    """
    policy.deterministic = True
    if hasattr(env, 'eval'):
        env.eval(env_eval)
    scores = [np.sum(gym_rollout_from_policy(env, policy, horizon)['rewards'])
              for _ in range(num_eps)]
    if hasattr(env, 'eval'):
        env.eval(False)
    return np.mean(scores), np.std(scores)


def unnormalize_action(
    env: gym.Env,
    action: np.ndarray,
) -> np.ndarray:
    """Convert action from between -1 and 1 to whatever the gym action bounds are.

    Args:
        env: The gym environment.
        action: The action between -1 and 1.

    Returns: The action in the true action space.
    """
    act_space = env.action_space
    act_spread = act_space.high - act_space.low
    return (action + 1) / 2 * act_spread + act_space.low


###########################################################################
#                           GYM IMPORT UTILS                              #
###########################################################################
D4RL_ENVIRONMENTS = set([])
for env_name in ['halfcheetah', 'walker2D', 'hopper', 'ant']:
    for dset in ['random', 'medium', 'expert', 'medium_replay', 'full_replay',
                 'medium_expert']:
        for version in ['v0', 'v1', 'v2']:
            D4RL_ENVIRONMENTS.add(f'{env_name}_{dset}-{version}')
PYBULLET_ENVIRONMENTS = set([])
# Add the occluded environments.
for env_name in ['HopperBLT', 'WalkerBLT', 'HalfCheetahBLT']:
    for obs_type in ['P', 'V', 'F']:
        PYBULLET_ENVIRONMENTS.add('-'.join([env_name, obs_type, 'v0']))


def extra_imports_for_env(env_name: str):
    """Make any additional imports needed for an environment.

    Better to isolate these imports because they should be optional.
    """
    if env_name in D4RL_ENVIRONMENTS:
        import d4rl
    if env_name in PYBULLET_ENVIRONMENTS:
        import dynamics_toolbox.rl.envs.pybullet
