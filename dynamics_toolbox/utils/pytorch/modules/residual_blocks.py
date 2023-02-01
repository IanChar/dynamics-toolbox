"""
Module that repeats specified block several times with residual connections.

Author: Ian Char
Date: February 1, 2023
"""
import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn


class ResidualBlocks(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_blocks: int,
        block_cfg: DictConfig,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            num_blocks: The number of blocks.
            block_cfg: The configuration of each of the blocks.
        """
        super().__init__()
        self.num_blocks = num_blocks
        for block_num in range(num_blocks):
            setattr(self, f'block_{block_num}', hydra.utils.instantiate(
                block_cfg,
                input_dim=input_dim,
                output_dim=output_dim,
                _recursive_=False,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for network

        Args:
            x: The input to the network.

        Returns:
            The output of the network.
        """
        for block_num in range(self.num_blocks):
            x = x + getattr(self, f'block_{block_num}')(x)
        return x
