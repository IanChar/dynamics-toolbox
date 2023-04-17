"""
Train an RL algorithm.

Author: Ian Char
Date: April 10, 2023
"""
import os

import hydra
from omegaconf import DictConfig, open_dict

from dynamics_toolbox.rl.util.gym_util import extra_imports_for_env
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm

import d4rl


@hydra.main(config_path='./example_configs/rl', config_name='online_sac_mujoco')
def train_rl(cfg: DictConfig):
    # Instantiate the gym environment and get the obs and act dims.
    extra_imports_for_env(cfg['env']['id'])
    env = hydra.utils.instantiate(cfg['env'])
    obs_dim = env.observation_space.low.shape[0]
    act_dim = env.action_space.low.shape[0]
    update_cfgs_with_dims(cfg, obs_dim, act_dim)
    # Set the device.
    dm.set_cuda_device(cfg.get('cuda_device', None))
    # Instantiate the RL algorithm.
    algorithm = hydra.utils.instantiate(cfg['algorithm'])
    # Run the train loop!
    hydra.utils.instantiate(
        cfg['train_loop'],
        algorithm=algorithm,
        env=env,
        logger={'run_dir': os.getcwd()},
        debug=cfg.get('debug', False),
    )


def update_cfgs_with_dims(cfg: DictConfig, obs_dim: int, act_dim: int) -> DictConfig:
    """Go through and every field that needs obs_dim and act_dim update."""
    required_keys = {'qnet', 'policy', 'replay_buffer', 'history_encoder',
                     'model_buffer', 'env_buffer'}
    for k, v in cfg.items():
        if k in required_keys:
            with open_dict(v):
                v['obs_dim'] = obs_dim
                v['act_dim'] = act_dim
        if isinstance(v, DictConfig):
            update_cfgs_with_dims(v, obs_dim, act_dim)


if __name__ == '__main__':
    train_rl()
