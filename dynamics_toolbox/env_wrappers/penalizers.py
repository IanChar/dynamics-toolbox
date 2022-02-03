"""
Penalty functions for model env.

Author: Ian Char
Date: 11/26/2021
"""


def std_width_penalizer(model_info: Dict[str, Any]) -> np.ndarray:
    """Penalize based on the width of std prediction.

    Args:
        model_info: Info outputted from model prediction.

    Returns:
        The penalty for each of the predictions.
    """
    return np.mean(model_info['std_predictions'], axis=-1).reshape(-1, 1)

def ensemble_max_std_width_penalizer(model_info: Dict[str, Any]) -> np.ndarray:
    """Penalize based on the width of max std prediction.

    Args:
        model_info: Info outputted from model prediction.

    Returns:
        The penalty for each of the predictions.
    """
    maxes = np.hstack([std_width_penalizer(memb) for memb in model_info.values()])
    return np.max(maxes, axis=1)

def ensemble_max_std_width_penalizer(model_info: Dict[str, Any]) -> np.ndarray:
    """Penalize based on standard deviation of ensemble members averaged across dim.

    Args:
        model_info: Info outputted from model prediction.

    Returns:
        The penalty for each of the predictions.
    """
    preds = np.array([memb['predictions'] for memb  in model_info.values()])
    return np.mean(np.std(preds, axis=0), axis=-1)

