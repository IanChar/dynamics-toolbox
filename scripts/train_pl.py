"""
Train a pytorch-lightning model.

Author: Ian Char
"""
from typing import NoReturn, Optional, List

from dynamics_toolbox.training import PL_TRAINING_TYPES
from dynamics_toolbox.utils.lightning.parsing import parse_pl_args


def run(remaining: Optional[List[str]] = None) -> NoReturn:
    """Train a pytorch lightning dynamics model.
    Args:
        remaining: The remaining strings after doing a previous parse.
    """
    args = parse_pl_args(remaining)
    PL_TRAINING_TYPES[args.training_type](args)


if __name__ == '__main__':
    run()
