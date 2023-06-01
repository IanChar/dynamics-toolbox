"""
Misc utilities for RL.

Author: Ian Char
Date: April 10, 2023
"""
import os
from typing import Optional

import hydra
from omegaconf import OmegaConf
import torch

from dynamics_toolbox.rl.modules.policies import Policy


def load_policy(path: str, checkpoint_epoch: Optional[int] = None) -> Policy:
    """Load in the policy.

    Args:
        path: The path to the RL run made.

    Returns: Policy.
    """
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    policy = hydra.utils.instantiate(cfg['algorithm']['policy'])
    policy_path = ('policy.pt' if checkpoint_epoch is None
                   else f'checkpoints/epoch_{checkpoint_epoch}.pt')
    policy.load_state_dict(torch.load(os.path.join(path, policy_path), 'cpu'))
    return policy


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
