"""
Cartpole with continuous actions.

Based on https://gist.github.com/iandanforth/e3ffb67cf3623153e968f2afdfb01dc8
"""
import math
from typing import Union, List

import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy
import torch


class CartPole(gym.Env):

    def __init__(
        self,
        dynamics_model=None,
        uncertainty_penalty_coef=0.0,
        **kwargs
    ):
        super().__init__()
        self._gravity = 9.8
        self._cart_mass = 1.0
        self._pole_mass = 0.1
        self._total_mass = self._pole_mass + self._cart_mass
        self._pole_length = 0.5
        self._pole_mass_length = self._pole_mass * self._pole_length
        self._force_mag = 30.0
        self._dt = 0.02
        self._theta_boundary = 12 * 2 * math.pi / 360
        self._x_boundary = 2.4
        obs_boundary = np.array([
            self._x_boundary * 2,
            np.finfo(np.float32).max,
            self._theta_boundary * 2,
            np.finfo(np.float32).max,
        ])
        self.observation_space = gym.spaces.Box(-obs_boundary, obs_boundary)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,))
        self.state = None
        self.model = dynamics_model
        self._uncertainty_penalty_coef = uncertainty_penalty_coef
        self._eval_mode = False

        ### BEGIN: adding beta mixture sampling code
        # To use bmp, we need in the kwargs:
        # - beta_mixture_process: True
        # - beta_mixture_params_dir: path to the directory containing the mixture params
        # - bmp_seed: seed for the random number generator

        if "beta_mixture_process" in kwargs and kwargs["beta_mixture_process"]:
            assert "beta_mixture_params_dir" in kwargs

            import scipy
            import torch
            import pickle as pkl
            import os

            input_dim = self.observation_space.shape[0] + self.action_space.shape[0]
            output_dim = self.observation_space.shape[0]
            from autocal.models.beta_mixture_process import BetaMixtureProcess
            self.beta_mixture_process = []

            num_ens_members = len(self.model.members)
            for ens_idx in range(num_ens_members):
                cur_ens_bmp = []
                for out_idx in range(output_dim):
                    mixture_params_path = os.path.join(
                        kwargs["beta_mixture_params_dir"],
                        f"{kwargs['bmp_seed']}-{ens_idx}",
                        "best_bmpt_params",
                        f"bmpt_dim{out_idx}.pkl")
                    print(f"LOADING BMPT: {mixture_params_path}")
                    beta_mixture_params = pkl.load(open(mixture_params_path, 'rb'))
                    kernel_lengthscales = beta_mixture_params['kls']
                    beta_concentration = beta_mixture_params['beta_const']
                    softmax_temp = beta_mixture_params['softmax_temp']
                    prior_kernel_weight = beta_mixture_params['prior_kernel_weight']
                    cur_dim_bmp = BetaMixtureProcess(
                        x_dim=input_dim,
                        kernel_lengthscales=kernel_lengthscales,
                        beta_concentration=beta_concentration,
                        softmax_temp=softmax_temp,
                        prior_kernel_weight=prior_kernel_weight)
                    cur_ens_bmp.append(cur_dim_bmp)
                self.beta_mixture_process.append(cur_ens_bmp)
        ### END: adding beta mixture sampling code

        if dynamics_model is None:
            self.transition = self._true_transition
        else:
            self.transition = self._model_transition

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=4)
        if self.model is not None:
            self.model.reset()
            ### BEGIN: adding beta mixture code
            if hasattr(self, "beta_mixture_process"):
                for ens_idx in range(len(self.beta_mixture_process)):
                    for out_dim_bmp in self.beta_mixture_process[ens_idx]:
                        out_dim_bmp.reset()
            ### END: adding beta mixture code
        return self.state, {}

    def eval(self, mode: bool = True):
        """Change eval mode."""
        self._eval_mode = mode
        if mode or self.model is None:
            self.transition = self._true_transition
        else:
            self.transition = self._model_transition

    def step(self, action):
        self.state, stats = self.transition(self.state, action)
        x, _, theta, _ = self.state
        done = bool(
            x < -self._x_boundary
            or x > self._x_boundary
            or theta < -self._theta_boundary
            or theta > self._theta_boundary
        )
        reward = float(not done)
        if self._uncertainty_penalty_coef > 0 and stats is not None:
            if 'std_predictions' in stats:
                # Do MOPO style penalty.
                if len(stats['std_predictions'].shape) > 2:
                    penalty = np.amax(np.linalg.norm(
                        stats['std_predictions']
                        * self._uncertainty_penalty_coef,
                        axis=-1), axis=0).reshape(-1, 1)
                else:
                    penalty = np.linalg.norm(stats['std_predictions']
                                             * self._uncertainty_penalty_coef,
                                             axis=-1).reshape(-1, 1)
            else:
                # Do disagreement based penalty.
                assert len(stats['predictions'].shape) > 2
                penalty = np.linalg.norm(
                    np.std(stats['predictions'], axis=0),
                    axis=-1).reshape(-1, 1)
            penalty = float(penalty)
            reward -= penalty
        return self.state, reward, done, {}, {}

    def _true_transition(self, s, a):
        force = self._force_mag * float(a)
        x, x_dot, theta, theta_dot = s
        sintheta, costheta = np.sin(theta), np.cos(theta)
        temp = ((force + self._pole_length * theta_dot ** 2 * sintheta)
                / self._total_mass)
        theta_acc = ((self._gravity * sintheta - costheta * temp) /
                     (self._pole_length
                      * (4/3 - self._pole_mass * costheta ** 2 / self._total_mass)))
        x_acc = temp - self._pole_mass_length * theta_acc * costheta / self._total_mass
        return np.array([
            x + x_dot * self._dt,
            x_dot + x_acc * self._dt,
            theta + theta_dot * self._dt,
            theta_dot + theta_acc * self._dt,
        ]), None

    def _model_transition(self, s, a):
        if not isinstance(a, np.ndarray):
            a = np.array([a])
        delta, info = self.model.predict(np.concatenate([s, a]).reshape(1, -1),
                                         each_input_is_different_sample=False)
        ### BEGIN: adding beta mixture code
        orig_delta = delta.copy()
        if hasattr(self, "beta_mixture_process"):
            # just need to get "pred"
            x = np.concatenate([s, a])
            # TODO: below only considers ensembles
            cur_use_ens_idx = self.model._curr_sample[0]
            # print(cur_use_ens_idx)
            mu = info['mean_predictions'][cur_use_ens_idx, 0]
            sigma = info['std_predictions'][cur_use_ens_idx, 0]
            # list of bmp for each output dim
            cur_use_bmp_list = self.beta_mixture_process[cur_use_ens_idx]
            x_normalized = self.model.normalizer.normalize(
                torch.from_numpy(
                    x[np.newaxis]).to(self.model._members[0].device),
                0).detach().cpu().numpy()
            next_quantile_levels = []
            for bmp in cur_use_bmp_list:
                cur_q = bmp.sample_next_q(x_normalized)
                next_quantile_levels.append(cur_q)
            # print([x.shape for x in next_quantile_levels])
            next_quantile_levels = np.stack(next_quantile_levels).reshape(*delta.shape)
            # next_quantile_levels = np.stack(
            #     [bmp.sample_next_q(x_normalized)
            #      for bmp in cur_use_bmp_list]).reshape(-1, output_dim)
            norm_sample = scipy.stats.norm(loc=mu, scale=sigma).ppf(
                next_quantile_levels)
            # TODO: below code for device only considers ensembles
            delta_sample = self.model._unnormalize_prediction_output(
                torch.from_numpy(norm_sample).to(
                    self.model._members[0].device))
            # pred = delta_sample.cpu().numpy().flatten() + x[:output_dim]
            delta = delta_sample.cpu().numpy().flatten().reshape(*delta.shape)
            print('orig_delta', orig_delta)
            print('delta', delta)
            breakpoint()
        ### END: adding beta mixture code

        return delta.flatten() + s, info

    @staticmethod
    def viz_trajectory(
        observations: Union[np.ndarray, List[np.ndarray]],
        actions: Union[np.ndarray, List[np.ndarray]],
    ):
        """Visualize the trajectory.

        Args:
            observations: The observations made for the trajectory with shape
                (N_trajs, time_steps, 4) or (time_steps, 4).
            actions: The action smade in the trajectory. With shape
                (N_trajs, time_steps, 1) or (time_steps, 1)
        """
        plt.style.use('seaborn')
        colors = ['blue', 'green', 'orange', 'purple']
        theta_boundary = 12 * 2 * math.pi / 360
        fig, axd = plt.subplot_mosaic([['x', 'th'],
                                       ['a', 'a']])
        if isinstance(observations, np.ndarray):
            observations = [observations]
            actions = [actions]
        num_trajs = len(observations)
        alpha = 10 / (9 + num_trajs)
        for tidx, ob in enumerate(observations):
            color = 'cornflowerblue' if num_trajs > len(colors) else colors[tidx]
            axd['x'].plot(ob[:, 0], color=color, alpha=alpha)
            axd['th'].plot(ob[:, 2], color=color, alpha=alpha)
            axd['a'].plot(actions[tidx].flatten())
        axd['x'].axhline(-2.4, color='red', ls=':')
        axd['x'].axhline(2.4, color='red', ls=':')
        axd['x'].set_xlabel('Time')
        axd['x'].set_ylabel('X Position')
        axd['x'].set_ylim([-2.5, 2.5])
        axd['th'].set_ylim([-theta_boundary - 0.1, theta_boundary + 0.1])
        axd['th'].axhline(-theta_boundary, color='red', ls=':')
        axd['th'].axhline(theta_boundary, color='red', ls=':')
        axd['th'].set_xlabel('Time')
        axd['th'].set_ylabel('Theta')
        axd['a'].set_xlabel('Time')
        axd['a'].set_ylabel('Force')
        plt.show()
