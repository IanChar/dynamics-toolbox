"""
Utilities for network initializations.

Author: Ian Char
Date: April 10, 2023
"""
import math

import torch.nn as nn


def init_default_mlp(m):
    """Iniatialize weights in a standard mlp according to
    https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073
    """
    if isinstance(m, nn.Linear):
        stdv = 1. / math.sqrt(m.weight.size(1))
        m.weight.data.uniform_(-stdv, stdv)
        if m.bias is not None:
            m.bias.data.uniform_(-stdv, stdv)
