"""
Main file to use for training dynamics models.

Author: Ian Char
"""
import os
import pickle as pkl

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities.seed import seed_everything

from dynamics_toolbox import DYNAMICS_TOOLBOX_PATH
from dynamics_toolbox.utils.lightning.constructors import\
        construct_all_pl_components_for_training


@hydra.main(config_path='./example_configs', config_name='example_rnn')
def train(cfg: DictConfig) -> None:
    """Train the model."""
    import torch
    torch.autograd.set_detect_anomaly(True)
    if cfg.get('debug', False):
        breakpoint()
    if 'model' not in cfg:
        raise ValueError('model must be specified. Choose one of the provided '
                         'model configurations and set +model=<model_config>.')
    if cfg['data_source'] == '':
        raise ValueError('data_source must be specified as either a path to an hdf5 '
                         'file or as a registered d4rl environment.')
    if 'seed' in cfg:
        seed_everything(cfg['seed'])
    # Alter config file and add defaults.
    if 'cuda_device' in cfg:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['cuda_device'])
    with open_dict(cfg):
        # Make the path relative.
        cfg['data_source'] = os.path.join(DYNAMICS_TOOLBOX_PATH, cfg['data_source'])
        cfg['data_module']['data_source'] = cfg['data_source']
        if 'smote' in cfg:
            cfg['smote'] = bool(cfg['smote'])
        cfg['save_dir'] = os.getcwd()
        if 'gpus' in cfg:
            cfg['gpus'] = str(cfg['gpus'])
    model, data, trainer, logger, cfg = construct_all_pl_components_for_training(cfg)
    print(OmegaConf.to_yaml(cfg))
    save_path = os.getcwd()
    OmegaConf.save(cfg, os.path.join(save_path, 'config.yaml'))
    logger_hparams = dict(cfg['model'], **cfg['data_module'])
    if 'dim_name_map' in logger_hparams:
        del logger_hparams['dim_name_map']
    logger.log_hyperparams(logger_hparams)
    trainer.fit(model, data)
    if data.val_dataloader() is not None:
        val_dict = trainer.validate(model, datamodule=data)
        with open('val_eval_stats.pkl', 'wb') as f:
            pkl.dump(val_dict, f)
    if data.test_dataloader() is not None:
        test_dict = trainer.test(model, datamodule=data)[0]
        with open('test_eval_stats.pkl', 'wb') as f:
            pkl.dump(test_dict, f)
        tune_metric = cfg.get('tune_metric', 'test/loss')
        return_val = test_dict[tune_metric]
        if cfg.get('tune_objective', 'minimize') == 'maximize':
            return_val *= -1
        return return_val
    else:
        return 0


if __name__ == '__main__':
    train()
