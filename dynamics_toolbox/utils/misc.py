"""
Misc helper functions.

Author: Ian Char
"""
from typing import Sequence, Optional, NoReturn, List, Union


def get_architecture(
        num_layers: Optional[int] = None,
        layer_size: Optional[int] = None,
        architecture: Optional[str] = None,
) -> Union[List[int], NoReturn]:
    if architecture is not None:
        hidden_sizes = s2i(architecture)
    elif num_layers is not None and layer_size is not None:
        hidden_sizes = [layer_size for _ in range(num_layers)]
    else:
        raise ValueError(
            'MLP architecture not provided. Either specify architecture '
            'argument or both num_layers and layer_size arguments.'
        )
    return hidden_sizes


def s2i(string: str) -> Sequence[int]:
    """Make a comma separated string of ints into a list of ints.

    Args:
        string: The string of csv ints.

    Returns: List of ints.
    """
    if '_' not in string:
        if len(string) > 0:
            return [int(string)]
        else:
            return []
    return [int(s) for s in string.split('_')]
