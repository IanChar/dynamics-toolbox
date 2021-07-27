"""
Gym environment that uses models.

Author: Ian Char
"""
from typing import Optional, Callable, Any, Dict, Tuple, Union, List

import gym
import numpy as np

from dynamics_toolbox.models.abstract_dynamics_model import AbstractDynamicsModel


class ModelEnv(gym.Env):

    def __init__(
            self,
            dynamics_model: AbstractDynamicsModel,
            start_distribution: Callable[[], np.ndarray],
            horizon: Optional[int] = None,
            penalizer: Optional[Callable[[Dict[str, Any]], float]] = None,
            penalty_coefficient: float = 1,
            terminal_function: Optional[Callable[[np.ndarray], bool]] = None,
            reward_function: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = None,
            reward_is_first_dim: bool = True,
            real_env: Optional[gym.Env] = None
    ):
        """
        Constructor.
        Args:
            dynamics_model: The model of the dynamics.
            start_distribution: Function that samples a start state.
            horizon: The horizon of the mdp. If not provided, the mdp has infinite horizon.
            penalizer: Function taking dictionary made by model and action and returning a penalty.
            penalty_coefficient: The coefficient to multiply the penalty by.
            terminal_function: Function taking next_state and returning whether termination has occurred.
            reward_function: A function taking state, action, next_state and returning
            reward_is_first_dim: Whether the first dimension of predictions from the model is rewards.
            real_env: The real environment being modelled.
        """
        super().__init__()
        if not reward_is_first_dim and reward_function is None:
            raise ValueError('Need a way to compute the reward.')
        self._dynamics = dynamics_model
        self._start_dist = start_distribution
        self._horizon = horizon
        self._penalizer = penalizer
        self._penalty_coefficient = penalty_coefficient
        self._terminal_function = terminal_function
        self._reward_function = reward_function
        self._reward_is_first_dim = reward_is_first_dim
        self._real_env = real_env
        self._t = 0
        self._state = self._start_dist()
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

    def reset(self) -> np.ndarray:
        """
        Reset the dynamics.
        Returns:
            The current observations.
        """
        self._t = 0
        self._state = self._start_dist()
        self._dynamics.reset()
        return self._state

    def step(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Make a step in the environment.
        Args:
            action: The action as a

        Returns:

        """
        if type(action) is float:
            action = np.array([action])
        nxt, model_info = self._dynamics.predict(self._state, action)
        self._t += 1
        info = {}
        rew, rew_info = self._compute_reward(self._state, action, nxt, model_info)
        info['raw_penalty'] = float(rew_info['raw_penalty'])
        # Compute terminal
        if self._terminal_function is not None:
            done = self._terminal_function(nxt)[0]
        else:
            done = False
        if self._horizon is not None:
            done = done or self._horizon >= self._t
        # Set nxt to current and return.
        self._state = nxt
        return nxt, rew, done, info

    def unroll_from_policy(
            self,
            starts: np.ndarray,
            policy: Callable[[np.ndarray], np.ndarray],
            horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unroll multiple different trajectories using a policy.
        Args:
            starts: All of the states to unroll from should have shape (num_states, obs_dim).
            policy: The policy taking state and mapping to action.
            horizon: The amount of time to unroll for.
        Returns:
            * All observations (num_starts, horizon + 1, obs_dim)
            * The actions taken (num_starts, horizon, act_dim)
            * The rewards received (num_starts, horizon)
            * The terminals (num_starts, horizon)
        """
        obs = np.zeros(starts.shape[0], horizon + 1, starts.shape[1])
        rewards = np.zeros(starts.shape[1], horizon)
        terminals = np.full((starts.shape[0], horizon), True)
        obs[:, 0, :] = starts
        acts = None
        for h in range(horizon):
            state = obs[:, h, :]
            act = policy(state)
            if acts is None:
                acts = np.zeros(starts.shape[0], horizon, act.shape[1])
                acts[:, h , :] = act
            nxts, infos = self._dynamics.predict(state, act)
            obs[:, h + 1, :] = nxts
            rewards[:, h] = self._compute_reward(state, act, nxts)
            if self._terminal_function is None:
                terminals[: h] = np.full(starts.shape[0], False)
            else:
                terminals[:, h] = np.array([self._terminal_function(nxt) for nxt in nxts])
        return obs, acts, rewards, terminals

    def unroll_from_actions(
            self,
            starts: np.ndarray,
            actions: np.ndarray,
            horizon: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Unroll multiple different trajectories using a policy.
        Args:
            starts: All of the states to unroll from should have shape (num_states, obs_dim).
            actions: The actions to use for unrolling should have shape (num_states, horizon, act_dim)>
            horizon: The amount of time to unroll for.
        Returns:
            * All observations (num_starts, horizon + 1, obs_dim)
            * The actions taken (num_starts, horizon, act_dim)
            * The rewards received (num_starts, horizon)
            * The terminals (num_starts, horizon)
        """
        act_idx = 0

        def policy_wrap(state: np.ndarray):
            return actions[:, act_idx, :]

        return self.unroll_from_policy(starts, policy_wrap, horizon)

    def render(self, mode='human'):
        """TODO: Figure out how to render given the real environment."""
        pass

    @property
    def t(self) -> int:
        """Get the time."""
        return self._t

    @property
    def state(self) -> np.ndarray:
        """Get the current state."""
        return self._state

    def _compute_reward(
            self,
            state: np.ndarray,
            action: Union[float, np.ndarray],
            nxt: np.ndarray,
            model_info: Union[Dict[str, Any], List[Dict[str, Any]]],
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
        rew = ...
        info = {}
        if self._reward_is_first_dim:
            rew = nxt[:, 0]
        if self._reward_function is not None:
            if len(state.shape) == 1:
                state = state.reshape(1, -1)
            if isinstance(action, float):
                action = np.array([[action]])
            if len(action.shape) == 1:
                action = action.reshape(1, -1)
            if len(nxt.shape) == 1:
                nxt = nxt.reshape(1, -1)
            rew = self._reward_function(state, action, nxt)
        if self._penalizer is not None:
            if isinstance(model_info, dict):
                model_info = [model_info]
            raw_penalty = np.array([self._penalizer(mi) for mi in model_info])
            rew -= self._penalty_coefficient * raw_penalty
            info['raw_penalty'] = raw_penalty
        return rew, info
