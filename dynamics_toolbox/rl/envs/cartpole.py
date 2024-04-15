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

        if dynamics_model is None:
            self.transition = self._true_transition
        else:
            self.transition = self._model_transition

    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=4)
        if self.model is not None:
            self.model.reset()
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
