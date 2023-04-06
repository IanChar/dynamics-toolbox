"""
Utilities for doing RL.

Author: Ian Char
Date: April 6, 2023
"""
import gym
from typing import Optional, Tuple

import numpy as np

from dynamics_toolbox.rl.policies.abstract_policy import Policy


def gym_rollout_from_policy(
    env: gym.Env,
    policy: Policy,
    horizon: Optional[int] = None,
) -> Tuple[np.ndarray]:
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
    obs.append(env.reset())
    policy.reset()
    running = True
    h = 0
    while running:
        act, logprob = policy.get_action(obs[-1])
        acts.append(act)
        logprobs.append(logprob)
        ob, rew, term, _ = env.step(act)
        obs.append(ob)
        rews.append(rew)
        terms.append(term)
        h += 1
        running = (not term) or ((horizon is not None) and h >= horizon)
    return (
        np.array(obs),
        np.array(acts),
        np.array(rews).reshape(-1, 1),
        np.array(terms).reshape(-1, 1),
        np.array(logprobs).reshape(-1, 1),
    )
