"""
Encode history using an RNN.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from dynamics_toolbox.rl.modules.history_encoders.abstract_history_encoder import (
    HistoryEncoder,
)


LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class RNNEncoder(HistoryEncoder):

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        rnn_type: str,
        rnn_hidden_size: int,
        obs_encode_dim: int,
        act_encode_dim: int,
        rew_encode_dim: int,
        rnn_num_layers: int = 1,
        encoder_activation=F.relu,
        **kwargs
    ):
        """Constructor.

        Args:
            obs_dim: Dimension of observation.
            act_dim: Dimension of action.
            rnn_type: Type of rnn to use.
            rnn_hidden_size: Size of rnn hidden unit.
            obs_encode_dim: Dimension of the observation encoding to make.
            act_encode_dim: Dimension of the action encoding to make.
            rew_encode_dim: Dimension of the reward encoding to make.
            rnn_num_layers: Number of layers in the rnn.
            encoder_activation: Hidden activation to use.
        """
        super().__init__()
        if obs_encode_dim <= 0:
            raise ValueError('Require obs encoder to be positive integer.')
        self.obs_encoder = nn.Linear(obs_dim, obs_encode_dim)
        if act_encode_dim > 0:
            self.act_encoder = nn.Linear(act_dim, act_encode_dim)
        else:
            self.act_encoder = None
        if rew_encode_dim > 0:
            self.rew_encoder = nn.Linear(1, rew_encode_dim)
        else:
            self.rew_encoder = None
        total_encode_dim = obs_encode_dim + act_encode_dim + rew_encode_dim
        self.rnn_type = rnn_type.lower()
        if rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {rnn_type}')
        self._memory_unit = rnn_class(total_encode_dim, rnn_hidden_size,
                                      num_layers=rnn_num_layers,
                                      batch_first=True)
        self._out_dim = rnn_hidden_size
        self.encoder_activation = encoder_activation
        # default gru initialization is uniform, not recommended
        # https://smerity.com/articles/2016/orthogonal_init.html
        # orthogonal has eigenvalue = 1 to prevent grad explosion or vanishing
        # Taken from Recurrent Networks are Strong Baseline for POMDP Repo.
        for name, param in self._memory_unit.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param)

    def forward(self, obs_seq, act_seq, rew_seq, history=None, encode_init=None):
        """Forward pass to get encodings.

        Args:
            obs_seq: Observations of shape (batch_size, seq length, obs dim)
            act_seq: Actions of shape (batch size, seq length, act dim)
            rew_seq: Rewards of sequence (batch size, seq length, 1)
            history: Previous encoding if this is here then we do not have to
                re-encode the full previous sequence and just need to take the last
                element of the sequence of each thing.
                If GRU expected shape is
                    (rnn_num_layers, batch_size, rnn_hidden_size)
                If LSTM expected tuple of two of the above.
            encoding_init: What to initialize the encoding at should have shape
                (batch_size, encode dim). This will not be used if history is
                provided. If LSTM we expect
                    (rnn_num_layers, batch_size, 2 * rnn_hidden_size)
                where hidden state and cell are concatted in last dimension and
                in that order.

        Returns:
            * Encodings of shape (batch size, seq length, out dim)
            * History of shape (num_layers, batch_size, rnn_hidden_size) and a tuple
              of these if it is an LSTM.
        """
        if history is not None:
            obs_seq, act_seq, rew_seq = [seq[:, [-1]]
                                         if seq is not None
                                         else None
                                         for seq in (obs_seq, act_seq, rew_seq)]
        elif encode_init is not None:
            history = encode_init.view(1, encode_init.shape[0], encode_init.shape[1])
            if self.rnn_type == 'lstm':
                split = history.shape[-1] // 2
                history = (history[..., :split], history[..., split:])
        obs_encoding = self.encoder_activation(self.obs_encoder(obs_seq))
        encoding = [obs_encoding]
        if self.act_encoder is not None:
            encoding.append(self.encoder_activation(self.act_encoder(act_seq)))
        if self.rew_encoder is not None:
            encoding.append(self.encoder_activation(self.rew_encoder(rew_seq)))
        encoding = torch.cat(encoding, dim=-1)
        encoding, new_history = self._memory_unit(encoding, history)
        return encoding, new_history

    @property
    def out_dim(self) -> int:
        """Output dimension of the encoding."""
        return self._out_dim

    @property
    def encoder_type(self) -> int:
        """Output dimension of the encoding."""
        return self.rnn_type
