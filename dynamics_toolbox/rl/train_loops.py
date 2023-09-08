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
        expl_returns = [np.sum(path['rewards']) for path in paths]
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
        stats_to_log = {}
        stats_to_log.update({f'{k}/mean': np.mean([d[k] for d in all_stats])
                             for k in all_stats[0].keys()})
        stats_to_log.update({f'{k}/std': np.std([d[k] for d in all_stats])
                             for k in all_stats[0].keys()})
        stats_to_log.update({f'{k}/min': np.min([d[k] for d in all_stats])
                             for k in all_stats[0].keys()})
        stats_to_log.update({f'{k}/max': np.min([d[k] for d in all_stats])
                             for k in all_stats[0].keys()})
        stats_to_log['ExplorationReturns/mean'] = np.mean(expl_returns)
        stats_to_log['ExplorationReturns/std'] = np.std(expl_returns)
        stats_to_log['ExplorationReturns/min'] = np.min(expl_returns)
        stats_to_log['ExplorationReturns/max'] = np.max(expl_returns)
        logger.log_epoch(
            epoch=ep,
            num_steps=num_steps_taken,
            stats=stats_to_log,
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
    reencode_buffer_every: int = -1,
    mask_tail_amount: float = 0.005,
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
        reencode_buffer_every: How often to reencode the buffer if using a history
            encoder. If this is set to 0 or below, no reencoding happens.
        mask_tail_amount: Percentage of extreme points to mask out for every
            observation dimension and reward.
        debug: Whether to set a breakpoint.
    """
    if debug:
        breakpoint()
    model_env.to(dm.device)
    num_steps_taken = 0
    eps_since_last_reencode = 0
    num_expl_paths_per_epoch = int(num_expl_paths_per_epoch)
    env_batch_size = int(batch_env_proportion * batch_size)
    model_batch_size = batch_size - env_batch_size
    model_env.start_dist = env_buffer.sample_starts
    if reencode_buffer_every > 0:
        _reencode_buffer(algorithm, env_buffer)
        model_buffer.encoding_dims = env_buffer.encoding_dims
        model_buffer.clear_buffer()
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
            mask_tail_amount=mask_tail_amount,
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
        # Possibly re-encode the buffer.
        if reencode_buffer_every > 0:
            logger.set_phase('Reencoding Buffer')
            if eps_since_last_reencode >= reencode_buffer_every:
                _reencode_buffer(algorithm, env_buffer)
                eps_since_last_reencode = 0
            else:
                eps_since_last_reencode += 1
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
        stats_to_log = {}
        # Log stats.
        stats_to_log.update({f'{k}/mean': np.mean([d[k] for d in all_stats])
                             for k in all_stats[0].keys()})
        stats_to_log.update({f'{k}/std': np.std([d[k] for d in all_stats])
                             for k in all_stats[0].keys()})
        stats_to_log.update({f'{k}/min': np.min([d[k] for d in all_stats])
                             for k in all_stats[0].keys()})
        stats_to_log.update({f'{k}/max': np.max([d[k] for d in all_stats])
                             for k in all_stats[0].keys()})
        # Log rollout stats.
        for info_stat in ('penalty', 'raw_reward'):
            if info_stat not in paths['info']:
                continue
            viable = np.concatenate([
                np.where(paths['masks'][:, idx, 0], ii, np.nan)
                for idx, ii in paths['info'][info_stat]
            ])
            stats_to_log['rollouts/{info_stat}/mean'] = np.nanmean(viable)
            stats_to_log['rollouts/{info_stat}/std'] = np.nanstd(viable)
            stats_to_log['rollouts/{info_stat}/min'] = np.nanmin(viable)
            stats_to_log['rollouts/{info_stat}/max'] = np.nanmax(viable)
        path_lengths = np.sum(paths['masks'], axis=1)
        stats_to_log['rollouts/path_length/mean'] = np.mean(path_lengths)
        stats_to_log['rollouts/path_length/std'] = np.std(path_lengths)
        stats_to_log['rollouts/path_length/min'] = np.min(path_lengths)
        stats_to_log['rollouts/path_length/max'] = np.max(path_lengths)
        logger.log_epoch(
            epoch=ep,
            num_steps=num_steps_taken,
            stats=stats_to_log,
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
        logger.start_inner_loop('Rollouts+Policy Updates', num_expl_steps_per_epoch)
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
            logger.end_inner_loop()
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


def _reencode_buffer(algorithm: RLAlgorithm, buffer: ReplayBuffer):
    if (hasattr(algorithm, 'get_history_encoders')
            and hasattr(buffer, 'reencode_paths')):
        buffer.reencode_paths(algorithm.get_history_encoders())
