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

_DATA_SOURCE_DIR = "/Users/youngsec/research/uq/highdim_cali/git_package/data/benchmark/mulan/mulan/data/multi-target/h5py_data"
_SAVE_DIR = "/Users/youngsec/research/uq/highdim_cali/git_package/experiments/benchmark"
# _DATA_SOURCE_DIR = "/zfsauton2/home/youngsec/research/uq/multidim_recal_paper/data/benchmark"
# _SAVE_DIR = "/zfsauton2/home/youngsec/research/uq/multidim_recal_paper/experiments/benchmark"
_CURR_DATASET_NAME = "???"
_CURR_SEED = 0

@hydra.main(config_path='./example_configs',
            config_name='config_benchmark_data')
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

    ######
    cfg['run_name'] = f"benchmark_dataset_{_CURR_DATASET_NAME}-seed_{_CURR_SEED}"
    cfg['data_source'] = f"{_DATA_SOURCE_DIR}/{_CURR_DATASET_NAME}-seed_{_CURR_SEED}.h5"
    cfg['save_dir'] = _SAVE_DIR

    print('='*80)
    print(f"  run_name: {cfg['run_name']}")
    print(f"  run_name: {cfg['data_source']}")
    print('='*80)
    ######



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
    model, data, trainer, logger, cfg = construct_all_pl_components_for_training(cfg)

    import pdb; pdb.set_trace()
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
    num_seeds = 10
    dataset_list = ["rf1", "rf2", "scm1d", "scm20d", "scpf"]
    # num_data_list = [500, 5000, 10000]
    dataset_list = ["rf1"]



    for dataset in dataset_list:
        for i in range(num_seeds):
            # cfg = OmegaConf.load('./example_configs/config_toy_data.yaml')
            # cfg['run_name'] = f"toy_dataset_{num_data}_rep{i}"
            # cfg['data_source'] = f"{_DATA_SOURCE_DIR}/dataset_{num_data}_rep{i}.h5"
            # cfg['save_dir'] = _SAVE_DIR
            global _CURR_DATASET_NAME
            _CURR_DATASET_NAME = dataset
            global _CURR_SEED
            _CURR_SEED = i
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

