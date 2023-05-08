"""
Misc utilities for RL.

Author: Ian Char
Date: April 10, 2023
"""


def soft_update_net(target_net, source_net, weight):
    """Update target net by doing a soft update from the source net.

    Based on code from rlkit https://github.com/rail-berkeley/rlkit

    Args:
        target_net: The network to udpate.
        source_net: The network to use for updating.
        weight: The convex combo weight.
    """
    for tparam, param in zip(target_net.parameters(), source_net.parameters()):
        tparam.data.copy_(
            tparam.data * (1.0 - weight) + param.data * weight
        )
