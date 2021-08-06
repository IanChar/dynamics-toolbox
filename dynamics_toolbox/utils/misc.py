"""
Misc helper functions.

Author: Ian Char
"""
from typing import Sequence


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
