"""
Training loop that does rollout(s) in the environment and then trains.

Author: Ian Char
Date: April 7, 2023
"""
import gym

import numpy as np

from dynamics_toolbox.env_wrappers.model_env import ModelEnv
from dynamics_toolbox.rl.algorithms.abstract_rl_algorithm import RLAlgorithm
from dynamics_toolbox.rl.buffers.abstract_buffer import ReplayBuffer
from dynamics_toolbox.rl.rl_logger import RLLogger
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm
from dynamics_toolbox.rl.util.gym_util import (
    explore_gym_until_threshold_met,
    evaluate_policy_in_gym,
)


def batch_online_rl_training(
    env: gym.Env,
    algorithm: RLAlgorithm,
    replay_buffer: ReplayBuffer,
    logger: RLLogger,
    epochs: int,
    num_expl_steps_per_epoch: int,
    num_eval_eps: int,
    horizon: int,
    num_gradient_steps_per_epoch: int,
    eval_frequency: int = 1,
    num_expl_steps_before_training: int = 0,
    batch_size: int = 256,
    debug: bool = False,
    **kwargs
):
    """Training loop that does rollout(s) in the environment then trains.

    Args:
        env: The environment.
        algorithm: The RL algorithm for training.
        replay_buffer: Replay buffer to store experience.
        logger: Logger for information.
        epochs: The number of epochs to run for.
        num_expl_steps_per_epoch: Number of exploration steps to take per epoch.
        num_eval_eps: Number of evaluation episodes to make per evaluation.
        horizon: Maximum steps taken for any given rollout.
        num_gradient_steps_per_epoch: Number of gradient steps to take every
            epoch.
        eval_frequency: How frequently we should do evaluation of policy.
        num_expl_steps_before_training: Number of exploration steps to warm
            start the buffer with.
        batch_size: Size of batch for each gradient step.
        debug: Whether to set a breakpoint.
    """
    if debug:
        breakpoint()
    num_steps_taken = 0
    # Possibly collect initial information.
    if num_expl_steps_before_training > 0:
        paths = explore_gym_until_threshold_met(
            env,
            algorithm.policy,
            num_steps=num_expl_steps_before_training,
            horizon=horizon,
        )
        for path in paths:
            replay_buffer.add_paths(path)
        num_steps_taken += num_expl_steps_before_training
    # Time to train!
    logger.start(epochs)
    for ep in range(epochs):
        # Collect.
        paths = explore_gym_until_threshold_met(
            env,
            algorithm.policy,
            num_steps=num_expl_steps_per_epoch,
            horizon=horizon,
        )
        for path in paths:
            replay_buffer.add_paths(path)
        num_steps_taken += num_expl_steps_per_epoch
        # Train.
        all_stats = []
        for _ in range(num_gradient_steps_per_epoch):
            batch = replay_buffer.sample_batch(batch_size)
            all_stats.append(algorithm.grad_step(batch))
        # Log.
        if ep % eval_frequency == 0:
            ret_mean, ret_std = evaluate_policy_in_gym(
                env,
                algorithm.policy,
                num_eval_eps,
                horizon=horizon,
            )
        else:
            ret_mean, ret_std = None, None
        logger.log_epoch(
            epoch=ep,
            num_steps=num_steps_taken,
            stats={k: np.mean([d[k] for d in all_stats]) for k in all_stats[0].keys()},
            returns_mean=ret_mean,
            returns_std=ret_std,
            policy=algorithm.policy,
        )
        replay_buffer.end_epoch()
    logger.end(algorithm.policy)


def mb_offline_rl_training(
    env: gym.Env,
    algorithm: RLAlgorithm,
    model_env: ModelEnv,
    model_buffer: ReplayBuffer,
    env_buffer: ReplayBuffer,
    logger: RLLogger,
    epochs: int,
    num_expl_paths_per_epoch: int,
    num_eval_eps: int,
    model_horizon: int,
    eval_horizon: int,
    num_gradient_steps_per_epoch: int,
    eval_frequency: int = 1,
    batch_size: int = 256,
    batch_env_proportion: float = 0.05,
    debug: bool = False,
    **kwargs
):
    """Train policy using a fixed, learned dynamics model.

    Args:
        env: The environment.
        algorithm: The RL algorithm for training.
        model_env: Environment wrapper for the learned dynamics model.
        model_buffer: Buffer of experience collected from the model.
        env_buffer: Buffer of the offline experience.
        logger: Logger for information.
        epochs: The number of epochs to run for.
        num_expl_paths_per_epoch: Number of exploration paths to take in the model
            per training epoch.
        num_eval_eps: Number of evaluation episodes to make per evaluation.
        model_horizon: The horizon for the model environment rollouts.
        eval_horizon: Horizon when doing evaluations.
        num_gradient_steps_per_epoch: Number of gradient steps to take every
            epoch.
        eval_frequency: How frequently we should do evaluation of policy.
        batch_size: Size of batch for each gradient step.
        batch_env_proportion: Proportion of the batch that should be made up of
            offline data.
        debug: Whether to set a breakpoint.
    """
    if debug:
        breakpoint()
    model_env.to(dm.device)
    num_steps_taken = 0
    num_expl_paths_per_epoch = int(num_expl_paths_per_epoch)
    env_batch_size = int(batch_env_proportion * batch_size)
    model_batch_size = batch_size - env_batch_size
    model_env.start_dist = env_buffer.sample_starts
    # Time to train!
    logger.start(epochs)
    for ep in range(epochs):
        # Collect.
        algorithm.policy.deterministic = False
        algorithm.policy.eval()
        paths = model_env.model_rollout_from_policy(
            num_rollouts=num_expl_paths_per_epoch,
            policy=algorithm.policy,
            horizon=model_horizon,
        )
        num_steps_taken += num_expl_paths_per_epoch * model_horizon
        model_buffer.add_paths(paths)
        # Train.
        all_stats = []
        for _ in range(num_gradient_steps_per_epoch):
            batch = model_buffer.sample_batch(model_batch_size)
            if env_batch_size > 0:
                env_batch = env_buffer.sample_batch(env_batch_size)
                batch = {k: np.concatenate([v, env_batch[k]], axis=0)
                         for k, v in batch.items()}
            all_stats.append(algorithm.grad_step(batch))
        # Log.
        if ep % eval_frequency == 0:
            ret_mean, ret_std = evaluate_policy_in_gym(
                env,
                algorithm.policy,
                num_eval_eps,
                horizon=eval_horizon,
            )
        else:
            ret_mean, ret_std = None, None
        logger.log_epoch(
            epoch=ep,
            num_steps=num_steps_taken,
            stats={k: np.mean([d[k] for d in all_stats]) for k in all_stats[0].keys()},
            returns_mean=ret_mean,
            returns_std=ret_std,
            policy=algorithm.policy,
        )
        model_buffer.end_epoch()
        env_buffer.end_epoch()
    logger.end(algorithm.policy)
