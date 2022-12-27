"""
Main file to use for training dynamics models.

Author: Ian Char
"""
import os
import pickle as pkl

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning.utilities.seed import seed_everything

import dynamics_toolbox
from dynamics_toolbox.utils.lightning.constructors import\
        construct_all_pl_components_for_training

_DATA_SOURCE_DIR = "/Users/youngsec/research/uq/highdim_cali/git_package/data/toy_dataset"
_SAVE_DIR = "/Users/youngsec/research/uq/highdim_cali/git_package/experiments/toy_dataset"
_CURR_NUM_DATA = 0
_CURR_REP_IDX = 0

@hydra.main(config_path='./example_configs',
            config_name='config_toy_data')
def train(cfg: DictConfig) -> None:
    """Train the model."""
    if 'model' not in cfg:
        raise ValueError('model must be specified. Choose one of the provided '
                         'model configurations and set +model=<model_config>.')
    if cfg['data_source'] == '':
        raise ValueError('data_source must be specified as either a path to an hdf5 '
                         'file or as a registered d4rl environment.')
    if 'seed' in cfg:
        seed_everything(cfg['seed'])

    ###
    cfg['run_name'] = f"toy_dataset_{_CURR_NUM_DATA}_rep{_CURR_REP_IDX}"
    cfg['data_source'] = f"{_DATA_SOURCE_DIR}/dataset_{_CURR_NUM_DATA}_rep{_CURR_REP_IDX}.h5"
    cfg['save_dir'] = _SAVE_DIR
    ###
    # Alter config file and add defaults.
    with open_dict(cfg):
        cfg['data_module']['data_source'] = cfg['data_source']
        if 'smote' in cfg:
            cfg['smote'] = bool(cfg['smote'])
        if 'save_dir' not in cfg:
            cfg['save_dir'] = os.path.join(os.getcwd(), 'model')
        elif cfg['save_dir'][0] != '/':
            cfg['save_dir'] = os.path.join(get_original_cwd(), cfg['save_dir'])
        if 'gpus' in cfg:
            cfg['gpus'] = str(cfg['gpus'])
    breakpoint()
    model, data, trainer, logger, cfg = construct_all_pl_components_for_training(cfg)

    # import pdb; pdb.set_trace()
    print(OmegaConf.to_yaml(cfg))
    if cfg['logger'] == 'mlflow':
        save_path = os.path.join(cfg['save_dir'], logger.experiment_id, logger.run_id)
    else:
        if 'run_name' in cfg:
            name = os.path.join(cfg['experiment_name'], cfg['run_name'])
        else:
            name = cfg['experiment_name']
        save_path = os.path.join(logger.save_dir, name, f'version_{logger.version}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    pkl.dump(data, open('{}/data_obj.pkl'.format(save_path), 'wb'))
    OmegaConf.save(cfg, os.path.join(save_path, 'config.yaml'))
    logger.log_hyperparams(dict(cfg['model'], **cfg['data_module']))
    # import pdb; pdb.set_trace()
    trainer.fit(model, data)
    if data.test_dataloader() is not None:
        test_dict = trainer.test(model, datamodule=data)[0]
        tune_metric = cfg.get('tune_metric', 'test/loss')
        return_val = test_dict[tune_metric]
        if cfg.get('tune_objective', 'minimize') == 'maximize':
            return_val *= -1
        return return_val
    else:
        return 0

def run():
    num_reps = 3
    num_data_list = [500, 1000, 3000]

    for num_data in num_data_list:
        for i in range(num_reps):
            # cfg = OmegaConf.load('./example_configs/config_toy_data.yaml')
            # cfg['run_name'] = f"toy_dataset_{num_data}_rep{i}"
            # cfg['data_source'] = f"{_DATA_SOURCE_DIR}/dataset_{num_data}_rep{i}.h5"
            # cfg['save_dir'] = _SAVE_DIR
            global _CURR_NUM_DATA
            _CURR_NUM_DATA = num_data
            global _CURR_REP_IDX
            _CURR_REP_IDX = i
            train()


if __name__ == '__main__':
    ### BEGIN: train script for toy data regression
    # num_reps = 10
    # num_data_list = [500, 1000, 3000, 5000, 8000, 10000]

    run()
    # num_reps = 3
    # num_data_list = [500, 1000, 3000]
    #
    # for num_data in num_data_list:
    #     for i in range(num_reps):
    #         # cfg = OmegaConf.load('./example_configs/config_toy_data.yaml')
    #         # cfg['run_name'] = f"toy_dataset_{num_data}_rep{i}"
    #         # cfg['data_source'] = f"{_DATA_SOURCE_DIR}/dataset_{num_data}_rep{i}.h5"
    #         # cfg['save_dir'] = _SAVE_DIR
    #         nonlocal _CURR_NUM_DATA
    #         _CURR_NUM_DATA = num_data
    #         nonlocal _CURR_REP_IDX
    #         _CURR_REP_IDX = i
    #         train()




    ### END: train script for toy data regression

    ### BEGIN: original code
    # train()
    ### END: original code

