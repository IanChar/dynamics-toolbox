"""
Training loop that does rollout(s) in the environment and then trains.

Author: Ian Char
Date: April 7, 2023
"""
import gym

import numpy as np

from dynamics_toolbox.rl.algorithms.abstract_rl_algorithm import RLAlgorithm
from dynamics_toolbox.rl.buffers.abstract_buffer import ReplayBuffer
from dynamics_toolbox.rl.rl_logger import RLLogger
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
):
    """Training loop that does rollout(s) in the environment then trains.

    Args:
        env: The environment.
        algorithm: The RL algorithm for training.
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
    """
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
    logger.end(algorithm.policy)
