"""
Penalty functions for model env.

Author: Ian Char
Date: 11/26/2021
"""
from typing import Any, Callable, Dict
import numpy as np


def get_penalizer(pen_name: str) -> Callable[Dict[str, Any], np.ndarray]:
    """Get penalizer function

    Args:
        pen_name: Name of the penalizer.

    Returns: Penalizer function.
    """
    if pen_name == 'std':
        return std_width_penalizer
    else:
        raise ValueError(f'Unknown penalizer {pen_name}')


def std_width_penalizer(model_info: Dict[str, Any]) -> np.ndarray:
    """Penalize based on the width of std prediction.

    Args:
        model_info: Info outputted from model prediction.

    Returns:
        The penalty for each of the predictions.
    """
    if len(model_info['std_predictions'].shape) > 2:
        return np.amax(np.linalg.norm(model_info['std_predictions'], axis=-1),
                       axis=0).reshape(-1, 1)
    else:
        return np.linalg.norm(model_info['std_predictions'], axis=-1).reshape(-1, 1)
