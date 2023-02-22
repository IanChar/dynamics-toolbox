"""
Different attention modules.

Author: Ian Char
Date: January 30, 2023
"""
import math
from typing import Callable

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Causal self attention. This is causal because we only attend on the past.
       Inspired by the code from Karpathy's nanoGPT repo.
    """

    def __init__(
        self,
        embed_dim_per_head: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """Constructor.

        Args:
            embed_dim: The dimension of the embedding per head.
            n_heads: The number of heads.
            block_size: Size of the block
            dropout: The amount of dropout.
            bias: Whether to have bias in the linear layer.
        """
        super().__init__()
        self.n_heads = n_heads
        self.embed_dim_per_head = embed_dim_per_head
        self.embed_dim = embed_dim_per_head * n_heads
        # This gives us the key, queries, and values for all heads.
        self.c_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(block_size, block_size)).view(1, 1,
                                                                block_size, block_size),
        )

    def forward(self, net_in: torch.Tensor) -> torch.Tensor:
        """Forward.

        Args:
            net_in: Input of shape (batch_size, sequence length, embed_dim)

        Returns: Output of size (batch_size, sequence length, embed_dim).
        """
        # B = batch_size, L = sequence length, C = embed_dim
        B, L, C = net_in.size()
        query, key, val = [vout.view(B, L, self.n_heads,
                                     self.embed_dim_per_head).transpose(1, 2)
                           for vout in self.c_proj(net_in).split(self.embed_dim, dim=2)]
        attn = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        attn = attn.masked_fill(self.bias[:, :, :L, :L] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        net_out = (attn @ val).transpose(1, 2).contiguous().view(B, L, C)
        return self.resid_dropout(self.out_proj(net_out))


class GPTBlock(nn.Module):

    def __init__(
        self,
        embed_dim_per_head: int,
        n_heads: int,
        block_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        hidden_activation: Callable[[torch.Tensor], torch.Tensor] = F.relu,
    ):
        """Constructor.

        Args:
            embed_dim: The dimension of the embedding per head.
            n_heads: The number of heads.
            block_size: Size of the block, i.e. the sequence length.
            dropout: The amount of dropout.
            bias: Whether to have bias in the linear layer.
        """
        super().__init__()
        embed_dim = embed_dim_per_head * n_heads
        self.ln1 = torch.nn.LayerNorm(embed_dim)
        self.ln2 = torch.nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim_per_head=embed_dim_per_head,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
            bias=bias,
        )
        self.c_fc = nn.Linear(embed_dim, 4 * embed_dim, bias=bias)
        self.c_proj = nn.Linear(4 * embed_dim, embed_dim, bias=bias)
        self.mlp_dropout = nn.Dropout(dropout)
        self.hidden_activation = hidden_activation

    def forward(self, net_in: torch.Tensor):
        """Forward pass.

        Args:
            net_in: Network input of shape (batch_size, sequence_length, embed dim)

        Returns: Network output of shape (batch_size, sequence_length, embed_dim)
        """
        net_in = net_in + self.attn(self.ln1(net_in))
        net_in = (net_in
                  + self.mlp_dropout(self.c_proj(self.hidden_activation(self.c_fc(
                    self.ln2(net_in))))))
        return net_in
