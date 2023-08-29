"""
Abstract class for modules that encoder RL history.

Author: Ian Char
Date: April 13, 2023
"""
import abc

import torch.nn as nn


class HistoryEncoder(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def forward(self, obs_seq, act_seq, rew_seq, history=None, encoding_init=None):
        """Forward pass to get encodings.

        Args:
            obs_seq: Observations of shape (batch_size, seq length, obs dim)
            act_seq: Actions of shape (batch size, seq length, act dim)
            rew_seq: Rewards of sequence (batch size, seq length, 1)
            history: Previous encoding if this is here then we do not have to
                re-encode the full previous sequence and just need to take the last
                element of the sequence of each thing.
            encoding_init: What to initialize the encoding at should have shape
                (batch_size, encode dim)

        Returns:
            * Encodings of shape (batch size, seq length, out dim)
            * History of shape (num_layers, batch_size, rnn_hidden_size) and a tuple
              of these if it is an LSTM.
        """

    @property
    @abc.abstractmethod
    def out_dim(self) -> int:
        """Output dimension of the encoding."""
