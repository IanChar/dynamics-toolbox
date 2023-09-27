"""
BetaN Tracking Environment.
"""
from typing import Any, Dict, Tuple, Optional, Union, List
import random

import gym
import numpy as np
import matplotlib.pyplot as plt


BETAN_MU = 1.7797029
BETAN_SIG = 1.2159256
PINJ_MU = 5817915.52734375
PINJ_SIG = 4223408.935546875
PINJ_ACT_SCALE = 0.25 * PINJ_SIG
TRANSITION_COEF = 2e2
BETA_OBS_COEF = 5.0
DT = 0.025


class BetaTracking(gym.Env):
    """Toy fusion environment.

    Obvervation Space: (BetaN, Curr Power, Target) (and possibly the PID components).
    Action Space: Change in the beam powers.
    Reward: -abs(P)
    """

    def __init__(
        self,
        w_start_dist: Tuple[float, float] = (5e4, 2.5e3),
        dw_start_dist: Tuple[float, float] = (0, 2.5e3),
        pinj_start_dist: Tuple[float, float] = (1.0e6, 1e5),
        aminor_dist: Tuple[float, float] = (0.589, 0.02),
        bt_dist: Tuple[float, float] = (2.75, 0.1),
        ip_dist: Tuple[float, float] = (1e6, 1e5),
        betan_target_bounds: Tuple[float, float] = (1.5, 2.25),
        pinj_bounds: Tuple[float, float] = (5e5, 1.2e7),
        w_bounds: Tuple[float, float] = (0.0, 5e5),
        dw_bounds: Tuple[float, float] = (-8.5e5, 8.5e5),
        momentum: float = 0.5,
        sample_parameters_every_episode: bool = False,
        fully_observable: bool = True,
        action_is_change: bool = True,
        include_pid_in_obs: bool = False,
    ):
        """Constructor.

        Args:
            include_pid_in_obs: Whether the observation space should have PID
                components included.
            w_start_dist: Mean and std deviation for the W start distribution.
            pinj_start_dist: Mean and std deviation for the Pinj start distribution.
            aminor_dist: Mean and std deviation for aminor.
            bt_dist: Mean and std deviation for bt.
            ip_dist: Mean and std deviation for ip.
            betan_target_bounds: Bounds on the betan target.
            pinj_bounds: Bounds on how high power can be.
            w_bounds: Bounds on the stored energy.
            dw_bounds: Bounds on how fast the stored energy can change.
            momentum: Added momentum term in the dynamics.
            obs_is_pid_only: Whether the observations should only be pid.
            action_is_change: Whether the action is change in current power or is it
                the delta based on the midpoint of the power bounds.
        """
        super().__init__()
        self.observation_dim = 4
        self.observation_space = gym.spaces.Box(
            -np.ones(self.observation_dim),
            np.ones(self.observation_dim),
        )
        self._action_is_change = action_is_change
        self.action_space = gym.spaces.Box(-1 * np.ones(1), np.ones(1))
        self.task_dim = 3  # i.e. aminor, bt, and ip.
        self._w_start_dist = w_start_dist
        self._dw_start_dist = dw_start_dist
        self._pinj_start_dist = pinj_start_dist
        self._aminor_dist = aminor_dist
        self._bt_dist = bt_dist
        self._ip_dist = ip_dist
        self._betan_target_bounds = betan_target_bounds
        self._pinj_bounds = pinj_bounds
        self._w_bounds = w_bounds
        self._dw_bounds = dw_bounds
        self._momentum = momentum
        self._sample_parameters_every_episode = sample_parameters_every_episode
        self._fully_observable = fully_observable
        self.state = None
        self._dt = 0.025
        self.reset_task()
        self._max_episode_steps = 100
        self._error_accum = None
        self.t = 0

    def reset(
        self,
        task: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Reset the system. If parameters are provided set them.

        Args:
            task: The task as [damping, stiffness, mass] with shape (3).
            targets: The target position to hit with shape (max_horizon)

        Returns:
            The observation.
        """
        self.t = 0
        self.reset_task(task)
        if targets is None:
            targets = self.sample_targets(1)[0]
        self.target = targets
        if start is None:
            start = self.sample_starts(1)[0]
        self.state = start
        self._error_accum = None
        return self._form_observation(self.state, self.target), {}

    def sample_starts(self, n_starts):
        start_ws = np.array([
            random.gauss(*self._w_start_dist)
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_dws = np.array([
            random.gauss(*self._dw_start_dist)
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        start_ps = np.array([
            np.clip(
                random.gauss(*self._pinj_start_dist),
                self._pinj_bounds[0],
                self._pinj_bounds[1]
            )
            for _ in range(n_starts)
        ]).reshape(-1, 1)
        return np.concatenate([
            start_ws,
            start_dws,
            start_ps,
        ], axis=1)

    def sample_tasks(self, n_tasks):
        if self._sample_parameters_every_episode:
            tasks = np.array([np.array([random.gauss(*self._aminor_dist),
                                        random.gauss(*self._bt_dist),
                                        random.gauss(*self._ip_dist)])
                              for _ in range(n_tasks)])
        else:
            tasks = np.array([np.array([self._aminor_dist[0],
                                        self._bt_dist[0],
                                        self._ip_dist[0]])
                              for _ in range(n_tasks)])
        return tasks

    def set_task(self, task):
        self._aminor = task[0]
        self._bt = task[1]
        self._ip = task[2]

    def get_task(self):
        return np.array([self._aminor, self._bt, self._ip])

    def reset_task(self, task=None):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, Dict[Any, Any]]:
        """"Take a step in the environment.

        Args:
            action: The force to applied.

        Returns:
            The next state, the reward, whether it is terminal and info dict.
        """
        self.t += 1
        if self._action_is_change:
            pinj_change = np.clip(action, -1, 1) * PINJ_ACT_SCALE
            next_pinj = float(np.clip(
                self.state[2] + pinj_change,
                self._pinj_bounds[0],
                self._pinj_bounds[1]
            ))
        else:
            midpoint = (self._pinj_bounds[1] - self._pinj_bounds[0]) / 2
            # next_pinj = np.clip(action, -1, 1) * midpoint + midpoint
            next_pinj = float(np.clip(
                np.clip(action, -1, 1) * midpoint + midpoint,
                self.state[2] - PINJ_ACT_SCALE,
                self.state[2] + PINJ_ACT_SCALE,
            ))
        next_dw = float(np.clip((self._momentum * self.state[1]
                                + (1 - self._momentum) * (next_pinj
                                - TRANSITION_COEF * self.state[0]
                                * self._ip ** -0.93
                                * self._bt ** -0.15
                                * self.state[2] ** 0.69)),
                                self._dw_bounds[0],
                                self._dw_bounds[1]))
        next_w = float(np.clip(
            self.state[0] + self._dt * self.state[1],
            self._w_bounds[0],
            self._w_bounds[1],
        ))
        next_state = np.array([next_w, next_dw, next_pinj]).flatten()
        obs, rew = self._form_observation_and_rew(
            next_state, self.target, None, self.state)
        self.state = next_state
        rew = np.abs(obs[0] - self.target)
        return obs, rew, False, {'target': self.target}, {}

    def rollout(
        self,
        policy,
        num_rollouts: int,
        horizon: int,
        task: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        start: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Do many rollouts at once following a policy. This should be much faster
           because we can vectorize the dynamics update.

        Args:
            policy: The policy to use for collections.
            num_rollouts: The number of rollouts to do.
            horizon: The length of the rollout.
            tasks: The tasks to roll forward with with shape (num rollouts, 3).
            targets: The tasks to roll forward with with shape
                (num rollouts, max_horizon).
            start_states: states to start at with shape (num_rollouts, 2)

        Returns dictionary with:
            - 'observations' (num_collects, horizon + 1, obs_dim)
            - 'actions' (num_collects, horizon, act_dim)
            - 'rewards' (num_collects, horizon)
            - 'terminals' (num_collects, horizon)
            - 'targets' (num_collects, horizon)
            - 'logpi': The log probabilities of actions (num_collects, horizon).
            - 'env_info': Information about the draws of the environment with
                shape (num_collects,).
        """
        self._error_accum = None
        policy.reset()
        # Draw system parameters and organize them into tensor.
        if task is None:
            task = self.sample_tasks(num_rollouts)
        assert len(task) == num_rollouts
        # Draw targets.
        if targets is None:
            targets = self.sample_targets(num_rollouts)
        assert len(targets) == num_rollouts
        # Initialize all of the data structures.
        obs = np.zeros((num_rollouts, horizon + 1, self.observation_dim))
        states = np.zeros((num_rollouts, horizon + 1, 3))
        acts = np.zeros((num_rollouts, horizon, 1))
        log_pis = np.zeros((num_rollouts, horizon))
        rewards = np.zeros((num_rollouts, horizon))
        terminals = np.full((num_rollouts, horizon), False)
        # Get initial states and observations.
        if start is None:
            start = self.sample_starts(num_rollouts)
        assert start.shape == (num_rollouts, 3)
        states[:, 0] = start
        obs[:, 0] = self._form_observation(states[:, 0], targets, task)
        for h in range(horizon):
            ob = obs[:, h]
            state = states[:, h]
            pi_act, log_pi = policy.get_actions(ob)
            acts[:, h] = pi_act
            log_pis[:, h] = log_pi
            if self._action_is_change:
                pinj_change = np.clip(pi_act, -1, 1) * PINJ_ACT_SCALE
                next_pinj = np.clip(
                    state[:, 2] + pinj_change.flatten(),
                    self._pinj_bounds[0],
                    self._pinj_bounds[1]
                )
            else:
                bound_radius = (self._pinj_bounds[1] - self._pinj_bounds[0]) / 2
                midpoint = bound_radius + self._pinj_bounds[0]
                next_pinj = np.clip(
                    np.clip(pi_act.flatten(), -1, 1) * bound_radius + midpoint,
                    state[:, 2] - PINJ_ACT_SCALE,
                    state[:, 2] + PINJ_ACT_SCALE,
                )
            next_dw = self._momentum * state[:, 1] + (1 - self._momentum) * np.clip(
                next_pinj
                - TRANSITION_COEF * state[:, 0]
                * task[:, 2] ** -0.93
                * task[:, 1] ** -0.15
                * state[:, 2] ** 0.69,
                self._dw_bounds[0],
                self._dw_bounds[1]
            )
            next_w = np.clip(
                state[:, 0] + self._dt * state[:, 1],
                self._w_bounds[0],
                self._w_bounds[1]
            )
            states[:, h + 1] = np.concatenate([
                next_w.reshape(-1, 1),
                next_dw.reshape(-1, 1),
                next_pinj.reshape(-1, 1),
            ], axis=1)
            obs[:, h + 1], rewards[:, h] = self._form_observation_and_rew(
                states[:, h + 1],
                targets,
                task,
                states[:, h]
            )
            if hasattr(policy, 'get_reward_feedback'):
                policy.get_reward_feedback(rewards[:, h])
        return {
            'observations': obs[:, :-1, :],
            'next_observations': obs[:, 1:, :],
            'actions': acts,
            'rewards': rewards[..., np.newaxis],
            'terminals': terminals[..., np.newaxis],
            'targets': targets,
            'logpi': log_pis,
            'actuators': states[..., -1],
        }

    def sample_targets(self, num_target_trajs: int) -> np.ndarray:
        """Draw targets.

        Args:
            num_target_trajs: The number of target trajectories.

        Returns: The target ndarray w shape (num_target_trajs, max horizon).
        """
        return np.array([random.uniform(*self._betan_target_bounds)
                         for _ in range(num_target_trajs)])

    def _form_observation(self, state, target, task=None, last_state=None):
        """Form the observation."""
        return self._form_observation_and_rew(state, target, task, last_state)[0]

    def _form_observation_and_rew(self, state, target, task=None, last_state=None):
        """Form the observation and the reward."""
        if task is None:
            task = self.get_task().reshape(1, -1)
        if not isinstance(target, np.ndarray):
            target = np.array([target])
        # Calculate the betan.
        to_convert = [state if len(state.shape) > 1 else state.reshape(1, -1)]
        if last_state is not None:
            to_convert.append(last_state if len(last_state.shape) > 1
                              else last_state.reshape(1, -1))
        betans = [
            tc[:, 0] * task[:, 0] * task[:, 1] / task[:, 2] * BETA_OBS_COEF
            for tc in to_convert
        ]
        betan_dws = [
            tc[:, 1] * task[:, 0] * task[:, 1] / task[:, 2] * BETA_OBS_COEF
            for tc in to_convert
        ]
        # Calculate the difference in the target.
        errs = betans[0] - target
        # Normalize terms in the observation and return.
        if self._fully_observable:
            if len(state.shape) > 1:
                raise NotImplementedError('TODO')
                obs = np.concatenate([
                    np.array([betans[0], betan_dws[0]]),
                    target.reshape(-1, 1),
                ], axis=1)
            else:
                obs = np.concatenate([betans[0], betan_dws[0], target])
        else:
            obs = np.concatenate([
                (betans[0].reshape(-1, 1) - BETAN_MU) / BETAN_SIG,
                (target.reshape(-1, 1) - BETAN_MU) / BETAN_SIG,
                (state[:, 2].reshape(-1, 1) - PINJ_MU) / PINJ_SIG,
            ], axis=1)
        return obs, -np.abs(errs) / BETAN_SIG

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
        if isinstance(observations, np.ndarray):
            observations = [observations]
            actions = [actions]
        num_trajs = len(observations)
        alpha = 10 / (9 + num_trajs)
        fig, axd = plt.subplot_mosaic([['w', 'p']])
        for tidx, ob in enumerate(observations):
            color = 'cornflowerblue' if num_trajs > len(colors) else colors[tidx]
            axd['w'].plot(ob[:, 0], color=color, alpha=alpha)
            axd['p'].plot(ob[:, -2], color=color, alpha=alpha)
        axd['w'].set_xlabel('Time')
        axd['w'].set_ylabel('Stored Energy')
        axd['p'].set_xlabel('Time')
        axd['p'].set_ylabel('Power')
        plt.show()
