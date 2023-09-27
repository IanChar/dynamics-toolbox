"""
Train an RL algorithm.

Author: Ian Char
Date: April 10, 2023
"""
import os

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from dynamics_toolbox.rl.util.gym_util import extra_imports_for_env
from dynamics_toolbox.utils.pytorch.device_utils import MANAGER as dm
import dynamics_toolbox.rl.envs


@hydra.main(config_path='./example_configs/rl', config_name='offline_mopo_d4rl')
def train_rl(cfg: DictConfig):
    # Instantiate the gym environment and get the obs and act dims.
    # extra_imports_for_env(cfg['env']['id'])
    env = hydra.utils.instantiate(cfg['env'])
    obs_dim = env.observation_space.low.shape[0]
    act_dim = env.action_space.low.shape[0]
    update_cfgs_with_dims(cfg, obs_dim, act_dim)
    # Set the device.
    dm.set_cuda_device(cfg.get('cuda_device', None))
    # Instantiate the RL algorithm.
    algorithm = hydra.utils.instantiate(cfg['algorithm'])
    # Save off the configuration.
    OmegaConf.save(cfg, 'config.yaml')
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
    input_output_keys = {'member_cfg'}
    for k, v in cfg.items():
        if k in required_keys:
            with open_dict(v):
                v['obs_dim'] = obs_dim
                v['act_dim'] = act_dim
        if k in input_output_keys:
            with open_dict(v):
                v['input_dim'] = obs_dim + act_dim
                v['output_dim'] = obs_dim + 1
        if isinstance(v, DictConfig):
            update_cfgs_with_dims(v, obs_dim, act_dim)


if __name__ == '__main__':
    train_rl()
