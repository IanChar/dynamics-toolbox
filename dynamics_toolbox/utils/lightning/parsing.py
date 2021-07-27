"""
Parsing utils for pytorch lightning.

Author: Ian Char
"""
import argparse
import os
from typing import List, Optional

import pytorch_lightning as pl

from dynamics_toolbox.configs import CONFIGS
from dynamics_toolbox.data.pl_data_modules.forward_dynamics import ForwardDynamicsDataModule
from dynamics_toolbox.models.pl_models import PL_MODELS
from dynamics_toolbox.training import PL_TRAINING_TYPES


def parse_pl_args(remaining: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse all the args needed for a pytorch-lightning model.
    Args:
        remaining: The remaining strings after doing a previous parse.
    Returns:
        The namespace from parsing.
    """
    # If a config is provided start with this.
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', choices=list(CONFIGS.keys()),
                               help='Name of configuration to use.')
    config_args, remaining = config_parser.parse_known_args(remaining)
    defaults = ...
    if config_args.config is not None:
        defaults = CONFIGS[config_args.config]
    # Figure out the model and training type.
    if defaults is not ...:
        model_type = defaults['model_type']
    else:
        pre_parser = argparse.ArgumentParser(add_help=False)
        pre_parser.add_argument('--model_type', choices=list(PL_MODELS.keys()), required=True)
        pre_args, remaining = pre_parser.parse_known_args(remaining)
        model_type = pre_args.model_type
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('dynamics-toolbox')
    group.add_argument('run_id',
                       help='The run ID associated with the experiment.'
                            'This will be appended to the root dir path.')
    group.add_argument('--training_type', choices=list(PL_TRAINING_TYPES.keys()),
                       default='single_model',
                       help='The type of training procedure to perform.')
    group.add_argument('--num_ensemble_members', type=int, default=1,
                       help='Number of members to train if training an ensemble.')
    group = parser.add_argument_group('Early Stopping')
    group.add_argument('--patience', type=int,
                       help='Number of checks to val loss of no improvement before stopping.'
                            'If None, then do not early stop.')
    group.add_argument('--min_delta', type=float, default=0.0,
                       help='Minimum absolute change that counts for val loss.')
    group.add_argument('--pudb', action='store_true',
                       help='Whether to activate pudb debugger.')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PL_MODELS[model_type].add_argparse_args(parser)
    parser = ForwardDynamicsDataModule.add_argparse_args(parser)
    if defaults is not ...:
        parser.set_defaults(**defaults)
    args = parser.parse_args(remaining)
    if args.data_source is None or not isinstance(args.data_source, str):
        raise ValueError('Need to specify --data_source.')
    if args.default_root_dir is None:
        args.default_root_dir = os.path.join(os.getcwd(), 'trained_models')
    args.default_root_dir = os.path.join(args.default_root_dir, args.run_id)
    if args.pudb:
        import pudb; pudb.set_trace()
    return args
