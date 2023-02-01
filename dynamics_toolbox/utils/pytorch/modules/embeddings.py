"""
Collection of different embeddings.

Author: Ian Char
Date January 31, 2023
"""
import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Taken from the pytorch transformers tutorial."""

    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        """Constructor.

        Args:
            embed_dim: The dimension of the embedding.
            dropout: The dropout rate.
            max_len: Maximum length of the sequence.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float()
                             * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, net_in: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        net_in = net_in + self.pe[:net_in.size(0), :]
        return self.dropout(net_in)
