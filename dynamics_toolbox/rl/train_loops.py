"""
Training loop that does rollout(s) in the environment and then trains.

Author: Ian Char
Date: April 7, 2023
"""
import gym

import numpy as np

from dynamics_toolbox.env_wrappers.model_env import ModelEnv
from dynamics_toolbox.env_wrappers.horizon_scheduler import HorizonScheduler
from dynamics_toolbox.rl.algorithms.abstract_rl_algorithm import RLAlgorithm
from dynamics_toolbox.rl.buffers.abstract_buffer import ReplayBuffer
from dynamics_toolbox.rl.dynamics_trainer import DynamicsTrainer
from dynamics_toolbox.rl.rl_logger import RLLogger
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm
from dynamics_toolbox.rl.util.gym_util import (
    explore_gym_until_threshold_met,
    evaluate_policy_in_gym,
    unnormalize_action,
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
        logger.set_phase('Environment Rollouts')
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
        logger.set_phase('Policy Updates')
        all_stats = []
        for _ in range(num_gradient_steps_per_epoch):
            batch = replay_buffer.sample_batch(batch_size)
            all_stats.append(algorithm.grad_step(batch))
        # Log.
        if ep % eval_frequency == 0:
            logger.set_phase('Policy Evaluation')
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


def offline_mbrl_training(
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
        logger.set_phase('Model Rollouts')
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
        logger.set_phase('Policy Updates')
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
            logger.set_phase('Policy Evaluation')
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


def online_mbrl_training(
    env: gym.Env,
    algorithm: RLAlgorithm,
    model_env: ModelEnv,
    model_buffer: ReplayBuffer,
    env_buffer: ReplayBuffer,
    logger: RLLogger,
    dynamics_trainer: DynamicsTrainer,
    epochs: int,
    num_expl_steps_per_epoch: int,
    num_model_paths_per_step: int,
    num_eval_eps: int,
    horizon_scheduler: HorizonScheduler,
    env_horizon: int,
    num_gradient_steps_per_step: int,
    eval_frequency: int = 1,
    batch_size: int = 256,
    batch_env_proportion: float = 0.05,
    num_expl_steps_before_training: int = 1000,
    model_val_proportion: float = 0.1,
    debug: bool = False,
    **kwargs
):
    """Train policy online learning a dynamics model where the model generates
       experience for the policy.

    Args:
        env: The environment.
        algorithm: The RL algorithm for training.
        model_env: Environment wrapper for the learned dynamics model.
        model_buffer: Buffer of experience collected from the model.
        env_buffer: Buffer of the offline experience.
        logger: Logger for information.
        epochs: The number of epochs to run for.
        num_expl_steps_per_epoch: Number of exploration steps to take in the real
            environment per epoch.
        num_model_paths_per_step: Number of model rollouts to do per step in
            the environment.
        num_eval_eps: Number of evaluation episodes to make per evaluation.
        horizon_scheduler: Schedule for the horizon.
        env_horizon: Horizon in the environment.
        num_gradient_steps_per_step: Number of gradient steps to take every
            epoch for the policy.
        eval_frequency: How frequently we should do evaluation of policy.
        batch_size: Size of batch for each gradient step.
        batch_env_proportion: Proportion of the batch that should be made up of
            offline data.
        model_val_proportion: The amount of data to use for validation when training
            the model.
        debug: Whether to set a breakpoint.
    """
    if debug:
        breakpoint()
    model_env.to(dm.device)
    num_steps_taken = 0
    num_expl_steps_per_epoch = int(num_expl_steps_per_epoch)
    env_batch_size = int(batch_env_proportion * batch_size)
    model_batch_size = batch_size - env_batch_size
    model_env.start_dist = env_buffer.sample_starts
    # Possibly collect initial information.
    if num_expl_steps_before_training > 0:
        paths = explore_gym_until_threshold_met(
            env,
            algorithm.policy,
            num_steps=num_expl_steps_before_training,
            horizon=env_horizon,
        )
        for path in paths:
            env_buffer.add_paths(path)
        num_steps_taken += num_expl_steps_before_training
    # Time to train!
    logger.start(epochs)
    env_t = 0
    curr_obs = env.reset()
    algorithm.policy.reset()
    for ep in range(epochs):
        # Train the dynamics models.
        logger.set_phase('Model Updates')
        curr_data_module = env_buffer.to_forward_dynamics_module(
            batch_size=batch_size,
            learn_rewards=True,  # TODO: Maybe change this later?
            val_proportion=model_val_proportion,
            num_workers=4,
        )
        model_dict = dynamics_trainer.fit(model_env.dynamics_model, curr_data_module)
        # Inner loop taking steps in the environment.
        algorithm.policy.deterministic = False
        algorithm.policy.eval()
        all_stats = []
        for _ in range(num_expl_steps_per_epoch):
            # Take a step in the environment.
            act, _ = algorithm.policy.get_actions(curr_obs)
            nxt, rew, term, _ = env.step(unnormalize_action(env, act))
            env_buffer.add_step(
                obs=curr_obs,
                nxt=nxt,
                act=act,
                rew=rew,
                terminal=term,
            )
            env_t += 1
            num_steps_taken += 1
            if env_t >= env_horizon or term:
                curr_obs = env.reset()
                env_t = 0
            else:
                curr_obs = nxt
            # Do several rollouts in the environment.
            logger.set_phase('Model Rollouts')
            for _ in range(num_model_paths_per_step):
                paths = model_env.model_rollout_from_policy(
                    num_rollouts=num_model_paths_per_step,
                    policy=algorithm.policy,
                    horizon=horizon_scheduler.get_horizon(num_steps_taken),
                )
                model_buffer.add_paths(paths)
            # Do updates to the policy.
            logger.set_phase('Policy Updates')
            for _ in range(num_gradient_steps_per_step):
                batch = model_buffer.sample_batch(model_batch_size)
                if env_batch_size > 0:
                    env_batch = env_buffer.sample_batch(env_batch_size)
                    batch = {k: np.concatenate([v, env_batch[k]], axis=0)
                             for k, v in batch.items()}
                all_stats.append(algorithm.grad_step(batch))
        # Log.
        if ep % eval_frequency == 0:
            logger.set_phase('Policy Evaluation')
            ret_mean, ret_std = evaluate_policy_in_gym(
                env,
                algorithm.policy,
                num_eval_eps,
                horizon=env_horizon,
            )
        else:
            ret_mean, ret_std = None, None
        stats = {k: np.mean([d[k] for d in all_stats]) for k in all_stats[0].keys()}
        stats.update(model_dict)
        logger.log_epoch(
            epoch=ep,
            num_steps=num_steps_taken,
            stats=stats,
            returns_mean=ret_mean,
            returns_std=ret_std,
            policy=algorithm.policy,
        )
        model_buffer.end_epoch()
        env_buffer.end_epoch()
    logger.end(algorithm.policy)
