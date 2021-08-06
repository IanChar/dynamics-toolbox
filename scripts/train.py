"""
Main file to use for training dynamics models.

Author: Ian Char
"""
import argparse
import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities.seed import seed_everything

import dynamics_toolbox
from dynamics_toolbox.utils.lightning.constructors import\
        construct_all_pl_components_for_training
from dynamics_toolbox.utils.storage.model_storage import save_config


@hydra.main(config_path=os.path.join(os.environ['DYNAMICS_TOOLBOX_PATH'], 'configs'),
            config_name='config')
def train(cfg: DictConfig) -> None:
    """Train the model."""
    if 'model' not in cfg:
        raise ValueError('model must be specified. Choose one of the provided '
                         'model configurations and set +model=<model_config>.')
    if cfg['data_source'] == '':
        raise ValueError('data_source must be specified as either a path to an hdf5 '
                         'file or as a registered d4rl environment.')
    seed_everything(cfg['seed'])
    with open_dict(cfg):
        cfg['data_module']['data_source'] = cfg['data_source']
        if 'save_dir' not in cfg:
            cfg['save_dir'] = os.path.join(get_original_cwd(), 'trained_models')
        elif cfg['save_dir'][0] != '/':
            cfg['save_dir'] = os.path.join(get_original_cwd(), cfg['save_dir'])
        if 'run_id' not in cfg:
            cfg['run_id'] = f"{cfg['data_source']}-{cfg['model']['model_name']}"
    if 'early_stopping' in cfg:
        with open_dict(cfg):
            cfg['early_stopping']['num_ensemble_members'] =\
                    cfg['model']['num_ensemble_members']
    cfg['early_stopping']
    model, data, trainer, logger, cfg = construct_all_pl_components_for_training(cfg)
    print(OmegaConf.to_yaml(cfg))
    save_config(trainer, cfg)
    logger.log_hyperparams(dict(cfg['model'], **cfg['data_module']))
    trainer.fit(model, data)
    test_dict = trainer.test(model, datamodule=data)[0]
    return test_dict['test/loss']


if __name__ == '__main__':
    train()

