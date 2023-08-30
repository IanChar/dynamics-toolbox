"""
Information about baselines.
"""
from collections import OrderedDict
import math

D4RL_REGULARIZATION = {
    'halfcheetah': (-280.178953, 12135.0),
    'hopper': (-20.272305, 3234.3),
    'walker2d': (1.629008, 4592.3),
}

D4RL_BASELINES = OrderedDict({
    'MAPLE': {
        'walker2d-random-v0': (21.07, 0.3 / math.sqrt(3)),
        'walker2d-medium-v0': (56.3, 10.6 / math.sqrt(3)),
        'walker2d-medium-replay-v0': (76.7, 3.8 / math.sqrt(3)),
        'walker2d-medium-expert-v0': (73.8, 8.0 / math.sqrt(3)),
        'halfcheetah-random-v0': (38.4, 1.3 / math.sqrt(3)),
        'halfcheetah-medium-v0': (50.4, 1.9 / math.sqrt(3)),
        'halfcheetah-medium-replay-v0': (59.0, 0.6 / math.sqrt(3)),
        'halfcheetah-medium-expert-v0': (63.5, 6.5 / math.sqrt(3)),
        'hopper-random-v0': (10.6, 0.1 / math.sqrt(3)),
        'hopper-medium-v0': (21.1, 1.2 / math.sqrt(3)),
        'hopper-medium-replay-v0': (87.5, 10.8 / math.sqrt(3)),
        'hopper-medium-expert-v0': (42.5, 4.1 / math.sqrt(3)),
    },
    'MOPO': {
        'walker2d-random-v0': (13.6, 2.6 / math.sqrt(3)),
        'walker2d-medium-v0': (11.8, 19.3 / math.sqrt(3)),
        'walker2d-medium-replay-v0': (39.0, 9.6 / math.sqrt(3)),
        'walker2d-medium-expert-v0': (44.6, 12.9 / math.sqrt(3)),
        'halfcheetah-random-v0': (35.4, 1.5 / math.sqrt(3)),
        'halfcheetah-medium-v0': (42.3, 1.6 / math.sqrt(3)),
        'halfcheetah-medium-replay-v0': (53.1, 2.0 / math.sqrt(3)),
        'halfcheetah-medium-expert-v0': (63.3, 38.0 / math.sqrt(3)),
        'hopper-random-v0': (11.7, 0.4 / math.sqrt(3)),
        'hopper-medium-v0': (28.0, 12.4 / math.sqrt(3)),
        'hopper-medium-replay-v0': (67.5, 24.7 / math.sqrt(3)),
        'hopper-medium-expert-v0': (23.7, 6.0 / math.sqrt(3)),
    },
})


def d4rl_normalize_and_get_baselines(
    data_path: str,
    avgs,
    errs,
):
    """Normalize results and return list of baselines.
    This relies on the environment name being somewhere in the path.

    Args:
        data_path: Name of the path where the data lives.
        avgs: The average scores as ndarray.
        errs: The standard errors of the scores as ndarray.

    Returns: Normalized avgs and errors and a dictionary of baselines.
    """
    baselines = {}
    # Check if we are dealing with a D4RL environment.
    env_types = ['halfcheetah', 'hopper', 'walker2d']
    data_types = ['random', 'medium', 'medium-replay', 'medium-expert', 'expert']
    version = ['v0', 'v2']
    d4rl_key, d4rl_env = None, None
    for et in env_types:
        for dt in data_types:
            for vt in version:
                if f'{et}-{dt}-{vt}' in data_path:
                    d4rl_key = f'{et}-{dt}-{vt}'
                    d4rl_env = et
                    break
    if d4rl_key is None:
        return avgs, errs, baselines
    low, high = D4RL_REGULARIZATION[d4rl_env]
    avgs = (avgs - low) / (high - low) * 100
    errs = errs / (high - low) * 100
    for k, v in D4RL_BASELINES.items():
        if d4rl_key in v:
            baselines[k] = v[d4rl_key]
    return avgs, errs, baselines
