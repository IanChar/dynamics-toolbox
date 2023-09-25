"""
Standard tanh Gaussian policy.

Author: Rohan Shah
Date: September 22, 2023
"""
from typing import Callable, Optional, Tuple, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import colorednoise
from tqdm import trange

from dynamics_toolbox.rl.modules.policies.abstract_policy import Policy
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm
from dynamics_toolbox.env_wrappers.model_env import ModelEnv

class MPCPolicy(Policy):
    """Generates actions using Model Predictive Control."""

    def __init__(
        self,
        model_env: ModelEnv,
        initial_variance_divisor: float = 0.5,
        base_nsamps: int = 100,
        planning_horizon: int = 10,
        n_elites: int = 10,
        beta: float = 0.25,
        gamma: float = 1.25,
        xi: float = 0.3,
        num_fs: int = 10,
        num_iters: int = 10,
        actions_per_plan: int = 1,
        actions_until_plan: int = 0,
        action_upper_bound: float = 1,
        action_lower_bound: float = -1,
        action_sequence: Optional[np.ndarray] = None,
    ):
        self.model_env = model_env
        self.obs_dim = model_env._observation_space.shape[0]
        self.action_dim = model_env._action_space.shape[0]
        self.initial_variance_divisor = initial_variance_divisor
        self.base_nsamps = base_nsamps
        self.planning_horizon = max(planning_horizon, 2)
        self.n_elites = n_elites
        self.beta = beta
        self.gamma = gamma
        self.xi = xi
        self.num_fs = num_fs
        self.num_iters = num_iters
        # self.verbose = getattr(, "verbose", False)
        self.actions_per_plan = actions_per_plan
        self.actions_until_plan = actions_until_plan
        self.action_sequence = action_sequence
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        # self.update_fn = update_fn
        # self.reward_fn = reward_fn
        # self.function_sample_list = getattr(, "function_sample_list", None)


    def get_initial_mean(self, action_sequence):
        if action_sequence is None:
            return np.zeros((self.planning_horizon, self.action_dim))
        else:
            new_action_sequence = np.concatenate(
                [action_sequence[1:, :], np.zeros((1, self.action_dim))], axis=0
            )
            return new_action_sequence[: self.planning_horizon, ...]
    

    def iCEM_generate_samples(
        self, nsamps, horizon, beta, mean, var, action_lower_bound, action_upper_bound
    ):
        action_dim = mean.shape[-1]
        samples = (
            colorednoise.powerlaw_psd_gaussian(
                beta, size=(nsamps, action_dim, horizon)
            ).transpose([0, 2, 1])
            * np.sqrt(var)
            + mean
        )
        samples = np.clip(samples, action_lower_bound, action_upper_bound)
        return samples        


    def get_actions(self, obs_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        horizon = self.planning_horizon
        beta = self.beta
        mean = self.get_initial_mean(self.action_sequence)
        if self.actions_until_plan > 0:
            self.action_sequence = mean
            action = mean[0, :]
            self.actions_until_plan -= 1
            return mean

        initial_variance_divisor = 4
        action_upper_bound = self.action_upper_bound
        action_lower_bound = self.action_lower_bound
        var = (
            np.ones_like(mean)
            * ((action_upper_bound - action_lower_bound) / initial_variance_divisor)
            ** 2
        )

        elites, elite_returns = None, None
        best_sample, best_return = None, -np.inf
        for i in trange(self.num_iters, disable=True):
            # these are num_samples x horizon x action_dim
            samples = self.iCEM_generate_samples(
                self.base_nsamps,
                horizon,
                beta,
                mean,
                var,
                action_lower_bound,
                action_upper_bound,
            )
            if i + 1 == self.num_iters:
                samples = np.concatenate([samples, mean[None, :]], axis=0)
                samples = samples[1:, ...]
            returns = self.evaluate_samples(obs_np, samples)
            if i > 0:
                elite_subset_idx = np.random.choice(
                    self.n_elites,
                    int(self.n_elites * self.xi),
                    replace=False,
                )
                elite_subset = elites[elite_subset_idx, ...]
                elite_return_subset = elite_returns[elite_subset_idx]
                samples = np.concatenate([samples, elite_subset], axis=0)
                returns = np.concatenate([returns, elite_return_subset])
            elite_idx = np.argsort(returns)[-self.n_elites :]
            elites = samples[elite_idx, ...]
            elite_returns = returns[elite_idx]
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)
            best_idx = np.argmax(returns)
            best_current_return = returns[best_idx]
            # logging.debug(f"{best_current_return=}")
            if best_current_return > best_return:
                best_return = best_current_return
                best_sample = samples[best_idx, ...]

        action = best_sample[0, :]
        self.action_sequence = best_sample
        # subtract one since we are already doing this action
        self.actions_until_plan = self.actions_per_plan - 1
        return action, None


    def evaluate_samples(self, obs_np, samples):
        """
        Evaluate the samples using the reward function.
        samples: (num_samples, horizon, action_dim)
        obs_np: (obs_dim,)
        """
        num_samples = samples.shape[0]
        # (num_samples, horizon, action_dim)
        samples = np.repeat(samples, self.num_fs, axis=0)
        # (num_samples * num_fs, horizon, action_dim)
        obs_np = np.repeat(obs_np[None, ...], num_samples * self.num_fs, axis=0)
        # (num_samples * num_fs, obs_dim)
        paths = self.model_env.model_rollout_from_actions(self.num_fs*num_samples, samples, obs_np, show_progress=False)
        rewards = paths['rewards']
        # (num_samples * num_fs, horizon)
        rewards = rewards.sum(axis=1)
        # (num_samples * num_fs,)
        rewards = rewards.reshape(num_samples, self.num_fs)
        # (num_samples, num_fs)
        returns = rewards.mean(axis=1)
        # (num_samples,)
        return returns


    def act_dim(self) -> int:
        return self.action_dim