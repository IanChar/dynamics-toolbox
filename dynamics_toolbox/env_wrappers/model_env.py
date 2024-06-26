"""
Gym environment that uses models.

Author: Ian Char
"""
from typing import Optional, Callable, Any, Dict, Tuple, Union

import gym
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from dynamics_toolbox.env_wrappers.bounders import Bounder
from dynamics_toolbox.rl.modules.policies.abstract_policy import Policy
from dynamics_toolbox.rl.modules.policies.action_plan_policy import ActionPlanPolicy
from dynamics_toolbox.models.abstract_model import AbstractModel
from dynamics_toolbox.models.ensemble import Ensemble


class ModelEnv(gym.Env):

    def __init__(
            self,
            dynamics_model: AbstractModel,
            start_distribution: Optional[Callable[[int], np.ndarray]] = None,
            horizon: Optional[int] = None,
            penalizer: Optional[Callable[[Dict[str, Any]], np.ndarray]] = None,
            penalty_coefficient: float = 1.0,
            terminal_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            reward_function: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray],
                                               np.ndarray]] = None,
            reward_is_first_dim: bool = True,
            real_env: Optional[gym.Env] = None,
            model_output_are_deltas: Optional[bool] = True,
            unscale_penalizer: bool = False,
            bounder: Optional[Bounder] = None,
    ):
        """
        Constructor.
        Args:
            dynamics_model: The model of the dynamics.
            start_distribution: Function that samples a start state.
            horizon: The horizon of the mdp. If not provided, the mdp has infinite
                horizon.
            penalizer: Function taking dictionary made by model and action and
                returning a penalty.
            penalty_coefficient: The coefficient to multiply the penalty by.
            terminal_function: Function taking next_state and returning whether
                termination has occurred.
            reward_function: A function taking state, action, next_state and returning
            reward_is_first_dim: Whether the first dimension of predictions from the
                model is rewards.
            real_env: The real environment being modelled.
            model_output_are_deltas: Whether the model predicts delta in state or the
                actual full state.
            unscale_penalizer: Whether to use unscaled uncertainty for penalty.
            bounder: Bounding for the states and rewards.
        """
        super().__init__()
        self._dynamics = dynamics_model
        self._start_dist = start_distribution
        self._horizon = horizon
        self._penalizer = penalizer
        self._bounder = bounder
        if not unscale_penalizer:
            self._std_scaling = 1
        elif (hasattr(self._dynamics, 'wrapped_model')
                and hasattr(self._dynamics.wrapped_model.normalizer, '1_scaling')):
            self._std_scaling =\
                getattr(self._dynamics.wrapped_model.normalizer,
                        '1_scaling').cpu().numpy()
        elif hasattr(self._dynamics.normalizer, '1_scaling'):
            self._std_scaling = getattr(self._dynamics.normalizer,
                                        '1_scaling').cpu().numpy()
        else:
            self._std_scaling = 1
        self._penalty_coefficient = penalty_coefficient
        self._terminal_function = terminal_function
        self._reward_function = reward_function
        self._reward_is_first_dim = reward_is_first_dim
        self._real_env = real_env
        self._model_output_are_deltas = model_output_are_deltas
        self._t = 0
        self._state = None
        if self._real_env is not None:
            self._observation_space = self._real_env.observation_space
            self._action_space = self._real_env.action_space
        else:
            obs_dim = self._dynamics.output_dim - self._reward_is_first_dim
            act_dim = self._dynamics.input_dim - obs_dim
            self._observation_space = gym.spaces.Box(
                low=-1 * np.ones(obs_dim),
                high=np.ones(obs_dim),
            )
            self._action_space = gym.spaces.Box(
                low=-1 * np.ones(act_dim),
                high=np.ones(act_dim),
            )

    def reset(self, start: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset the dynamics.

        Args:
            start: Start for the system.

        Returns:
            The current observations.
        """
        self._t = 0
        if start is not None:
            self._state = start
        elif self._start_dist is not None:
            self._state = self._start_dist(1).flatten()
        else:
            raise ValueError('Starts must be provided if start state dist is not.')
        self._dynamics.reset()
        return self._state

    def step(self, action: Union[float, np.ndarray]) \
            -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """ Make a step in the environment.

        Args:
            action: The action as a

        Returns:
            - The next state.
            - The reward for the transition.
            - Whether a terminal state was reached.
            - Extra information.
        """
        if self._state is None:
            raise RuntimeError('Must call reset before step.')
        if type(action) is float:
            action = np.array([action])
        if len(action.shape) == 1:
            action = action.reshape(1, -1)
        model_out, model_info = self._dynamics.predict(np.hstack(
            [self._state.reshape(1, -1), action]))
        nxt = (model_out + self._state.reshape(1, -1) if self._model_output_are_deltas
               else model_out)
        if self._bounder is not None:
            nxt = self._bounder.bound_state(nxt, self._state.reshape(1, -1))
        self._t += 1
        info = {}
        rew, rew_info = self._compute_reward(self._state, action, nxt, model_info)
        if 'penalty' in rew_info:
            info['penalty'] = float(rew_info['penalty'])
        # Compute terminal
        if self._terminal_function is not None:
            done = self._terminal_function(nxt)[0]
        else:
            done = False
        if self._horizon is not None:
            done = done or self._horizon >= self._t
        # Set nxt to current and return.
        self._state = nxt
        return nxt.flatten(), float(rew), done, info

    def model_rollout_from_policy(
            self,
            num_rollouts: int,
            policy: Policy,
            horizon: int,
            starts: Optional[np.ndarray] = None,
            start_info: Optional[Dict] = None,
            show_progress: bool = False,
            mask_tail_amount: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Unroll multiple different trajectories using a policy.
        Args:
            num_rollouts: The number of rollouts to be made.
            policy: The policy taking state and mapping to action.
            horizon: The amount of time to unroll for.
            starts: All of the states to unroll from should have shape
                (num_rollouts, obs_dim).
            show_progress: Whether to show the progress of the rollout.
            mask_tail_amount: Percentage of extreme points to mask out for every
                observation dimension and reward.
        Returns:
            - observations: All observations (num_rollouts, horizon + 1, obs_dim)
            - actions: The actions taken (num_rollouts, horizon, act_dim)
            - rewards: The rewards received (num_rollouts, horizon, 1)
            - terminals: The terminals (num_rollouts, horizon, 1)
            - logprobs: The logprobabilities of actions (num_rollouts, horizon, 1)
            - masks: Mask for whether the data is real or not. 0 if the transition
                happened after a terminal. Has shape (num_rollouts, horizon, 1)
        """
        if starts is None:
            if self._start_dist is None:
                raise ValueError('Starts must be provided if start state dist is not.')
            starts, start_info = self._start_dist(num_rollouts)
        else:
            assert len(starts) == num_rollouts, 'Number of starts must match.'
        if start_info is not None:
            pi_encoding = start_info.get('pi_encoding', None)
        else:
            pi_encoding = None
        policy.reset(init_encoding=pi_encoding)
        self._dynamics.reset()
        obs = np.zeros((starts.shape[0], horizon + 1, starts.shape[1]))
        rews = np.zeros((starts.shape[0], horizon, 1))
        terms = np.full((starts.shape[0], horizon, 1), True)
        logprobs = np.zeros((starts.shape[0], horizon, 1))
        masks = np.ones((starts.shape[0], horizon, 1))
        all_infos = []
        obs[:, 0, :] = starts
        acts = None
        if show_progress:
            pbar = tqdm(total=horizon)
        for h in range(horizon):
            state = obs[:, h, :]
            act, logprob = policy.get_actions(state)
            logprobs[:, h] = logprob.reshape(-1, 1)
            if acts is None:
                acts = np.zeros((starts.shape[0], horizon, act.shape[1]))
            acts[:, h, :] = act
            model_out, infos = self._dynamics.predict(np.hstack([state, act]))
            curr_rew, rew_info = self._compute_reward(state, act, model_out, infos)
            rews[:, h] = curr_rew
            infos.update(rew_info)
            all_infos.append(infos)
            if self._reward_is_first_dim:
                model_out = model_out[:, 1:]
            nxts = state + model_out if self._model_output_are_deltas else model_out
            if self._bounder is not None:
                nxts = self._bounder.bound_state(nxts, state)
            obs[:, h + 1, :] = nxts
            policy.get_reward_feedback(rews[:, h])
            if self._terminal_function is None:
                terms[:, h] = np.full(terms[:, h].shape, False)
            else:
                terms[:, h] = self._terminal_function(nxts).reshape(-1, 1)
                if np.sum(terms[:, h]) > 0:
                    term_idxs = np.argwhere(terms[:, h].flatten())
                    masks[term_idxs, h + 1:] = 0
            if show_progress:
                pbar.update(1)
        if show_progress:
            pbar.close()
        if mask_tail_amount > 0.0:
            for obdim in range(obs.shape[-1]):
                low, high = np.quantile(obs[..., obdim],
                                        [mask_tail_amount / 2,
                                         1 - (mask_tail_amount / 2)])
                extreme_idxs = np.argwhere(np.logical_or(
                    obs[..., obdim] < low, obs[..., obdim] > high))
                for etraj, eh in extreme_idxs:
                    masks[etraj, eh:] = 0
                    # TODO: Do we want to put a terminal before things go crazy?
                    terms[etraj, min(eh - 1, 0)] = 1
            low, high = np.quantile(rews, [mask_tail_amount / 2,
                                           1 - (mask_tail_amount / 2)])
            extreme_idxs = np.argwhere(np.logical_or(
                rews[..., 0] < low, rews[..., 0] > high))
            for etraj, eh in extreme_idxs:
                masks[etraj, eh:] = 0
                terms[etraj, min(eh - 1, 0)] = 1
        paths = {
            'observations': obs,
            'actions': acts,
            'rewards': rews,
            'terminals': terms,
            'logprobs': logprobs,
            'masks': masks,
            'info': all_infos,
        }
        # If the start came with an encoding, load this in as the first part of
        # the path. The rest is 0s since right now there is no way to extract
        # encoding as we roll out. TODO: Add a way to get these encodings. However,
        # right now rollout length is usually <= sequence buffer lookback so it
        # is not a problem.
        if start_info is not None:
            for k, v in start_info.items():
                if 'encoding' in k:
                    path_encoding = np.zeros((acts.shape[0],
                                              acts.shape[1], v.shape[-1]))
                    path_encoding[:, 0] = v
                    paths[k] = path_encoding
        return paths

    def model_rollout_from_actions(
            self,
            num_rollouts: int,
            actions: np.ndarray,
            starts: Optional[np.ndarray] = None,
            show_progress: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unroll multiple different trajectories using a policy.
        Args:
            actions: The actions to use for unrolling should have shape
                (num_states, horizon, act_dim)>
            starts: All of the states to unroll from should have shape
                (num_states, obs_dim).
                If not specified will be drawn from start state dist.
        Returns:
            - All observations (num_starts, horizon + 1, obs_dim)
            - The actions taken (num_starts, horizon, act_dim)
            - The rewards received (num_starts, horizon)
            - The terminals (num_starts, horizon)
            - The logprobabilities of the actions.
            - masks: Mask for whether the data is real or not. 0 if the transition
                happened after a terminal. Has shape (num_rollouts, horizon, 1)
        """
        horizon = actions.shape[1]
        policy_wrap = ActionPlanPolicy(actions)
        return self.model_rollout_from_policy(num_rollouts, policy_wrap, horizon,
                                              starts=starts,
                                              show_progress=show_progress)

    def render(self, mode='human'):
        """TODO: Figure out how to render given the real environment."""
        pass

    def to(self, device: str):
        """For any pytorch modules put on the device.

        Args:
            device: The device to put onto.
        """
        if isinstance(self._dynamics, Ensemble):
            for member in self._dynamics.members:
                if isinstance(member, nn.Module):
                    member.to(device)
        elif isinstance(self._dynamics, nn.Module):
            self._dynamics.to(device)
        if hasattr(self._dynamics, 'wrapped_model'):
            if isinstance(self._dynamics.wrapped_model, Ensemble):
                for member in self._dynamics.wrapped_model.members:
                    if isinstance(member, nn.Module):
                        member.to(device)
            elif isinstance(self._dynamics.wrapped_model, nn.Module):
                self._dynamics.wrapped_model.to(device)

    @property
    def t(self) -> int:
        """Get the time."""
        return self._t

    @property
    def state(self) -> np.ndarray:
        """Get the current state."""
        return self._state

    @property
    def start_dist(self) -> Callable[[int], np.ndarray]:
        return self._start_dist

    @property
    def dynamics_model(self) -> AbstractModel:
        return self._dynamics

    @start_dist.setter
    def start_dist(self, dist: Callable[[int], np.ndarray]):
        self._start_dist = dist

    def _compute_reward(
            self,
            state: np.ndarray,
            action: Union[float, np.ndarray],
            nxt: np.ndarray,
            model_info: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Compute the rewards.
        Args:
            state: The states.
            action: The actions
            nxt: The next states.
            model_info: The info outputted by the dynamics model.

        Returns:
            The corresponding rewards.
        """
        rew = np.zeros((len(state), 1))
        info = {}
        if self._reward_is_first_dim:
            rew = nxt[:, [0]]
        elif self._reward_function is not None:
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            if isinstance(action, float):
                action = np.array([[action]])
            if len(action.shape) == 1:
                action = action.reshape(1, -1)
            if len(nxt.shape) == 1:
                nxt = nxt.reshape(1, -1)
            rew = self._reward_function(state, action, nxt)
        if self._bounder is not None:
            rew = self._bounder.bound_reward(rew)
        info['raw_reward'] = rew
        if self._penalizer is not None:
            model_info['std_scaling'] = self._std_scaling
            penalty = self._penalizer(model_info)
            rew -= self._penalty_coefficient * penalty
            info['penalty'] = penalty
        return rew, info
