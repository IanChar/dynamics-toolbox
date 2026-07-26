"""
Transformer-based model that predicts a gaussian distribution.
This is designed as a drop-in replacement for RPNN, using transformer
attention mechanisms instead of RNN/GRU.

Author: AI Assistant
Date: 12/06/2024
"""
from typing import Dict, Callable, Tuple, Any, Sequence, Optional

import hydra.utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from dynamics_toolbox.constants import losses, sampling_modes
from dynamics_toolbox.models.pl_models.sequential_models.abstract_sequential_model \
        import AbstractSequentialModel
from dynamics_toolbox.utils.pytorch.losses import get_regression_loss
from dynamics_toolbox.utils.pytorch.metrics import SequentialExplainedVariance
from dynamics_toolbox.utils.pytorch.modules.attention import GPTBlock

from dynamics_toolbox.utils.storage.model_storage import (
    load_model_from_log_dir,
    load_ensemble_from_parent_dir,
)

import os
import numpy as np


class TPNN(AbstractSequentialModel):
    """Transformer Probabilistic Neural Network.
    
    This architecture uses self-attention mechanisms (transformer blocks) instead
    of RNN/GRU for sequence modeling. It maintains the same interface as RPNN
    for compatibility with existing pipelines.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            encode_dim: int,
            num_transformer_blocks: int,
            num_heads: int,
            block_size: int,
            encoder_cfg: DictConfig,
            pnn_decoder_cfg: DictConfig,
            dropout: float = 0.1,
            warm_up_period: int = 0,
            learning_rate: float = 1e-3,
            logvar_lower_bound: Optional[float] = None,
            logvar_upper_bound: Optional[float] = None,
            logvar_bound_loss_coef: float = 1e-3,
            sample_mode: str = sampling_modes.SAMPLE_FROM_DIST,
            weight_decay: Optional[float] = 0.0,
            use_layer_norm: bool = True,
            use_positional_encoding: bool = True,
            mask_indices: Optional[Sequence[int]] = [],
            loss_fn_str: str = "NLL",
            mse_wt: float = 1.0,
            nll_wt: float = 1.0,
            load_dir: Optional[str] = None,
            seed: Optional[int] = None,
            **kwargs,
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            encode_dim: The dimension of the encoder output.
            num_transformer_blocks: Number of transformer blocks.
            num_heads: Number of attention heads in transformer.
            block_size: Maximum sequence length for the transformer.
            encoder_cfg: The configuration for the encoder network.
            pnn_decoder_cfg: The configuration for the decoder network. Should
                be a PNN.
            dropout: Dropout probability for transformer blocks.
            warm_up_period: The amount of data to take in before predictions begin to
                be made.
            learning_rate: The learning rate for the network.
            logvar_lower_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_upper_bound: Lower bound on the log variance.
                If none there is no bound.
            logvar_bound_loss_coef: Coefficient on bound loss to add to loss.
            sample_mode: The method to use for sampling.
            weight_decay: The weight decay for the optimizer.
            use_layer_norm: Whether to use layer norm after encoding.
            use_positional_encoding: Whether to add positional encodings.
            mask_indices: The indices to mask in the input.
            loss_fn_str: Loss function string ("NLL", "MSE", or "NLL+MSE").
            mse_wt: Weight for MSE loss when using combined loss.
            nll_wt: Weight for NLL loss when using combined loss.
            load_dir: Directory to load pretrained weights from.
            seed: Random seed for loading specific checkpoint.
        """
        super().__init__(input_dim, output_dim, **kwargs)
        
        # Store configuration
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._encode_dim = encode_dim
        self._num_transformer_blocks = num_transformer_blocks
        self._num_heads = num_heads
        self._block_size = block_size
        self._warm_up_period = warm_up_period
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._sample_mode = sample_mode
        self._record_history = True
        self._use_layer_norm = use_layer_norm
        self._use_positional_encoding = use_positional_encoding
        
        # Build encoder
        self._encoder = hydra.utils.instantiate(
            encoder_cfg,
            input_dim=input_dim,
            output_dim=encode_dim,
            _recursive_=False,
        )
        
        # Layer norm after encoder
        if use_layer_norm:
            self._layer_norm = torch.nn.LayerNorm(encode_dim)
        
        # Positional encoding
        if use_positional_encoding:
            self._positional_encoding = nn.Parameter(
                torch.zeros(1, block_size, encode_dim)
            )
            nn.init.normal_(self._positional_encoding, mean=0.0, std=0.02)
        
        # Transformer blocks
        if encode_dim % num_heads != 0:
            raise ValueError(f"encode_dim ({encode_dim}) must be divisible by num_heads ({num_heads})")
        
        embed_dim_per_head = encode_dim // num_heads
        self._transformer_blocks = nn.ModuleList([
            GPTBlock(
                embed_dim_per_head=embed_dim_per_head,
                n_heads=num_heads,
                block_size=block_size,
                dropout=dropout,
                bias=True,
            ) for _ in range(num_transformer_blocks)
        ])
        
        # Decoder (PNN-style with mean and logvar heads)
        self._decoder = hydra.utils.instantiate(
            pnn_decoder_cfg,
            input_dim=encode_dim,
            output_dim=output_dim,
            _recursive_=False,
        )
        
        # Set up variance pinning
        self._var_pinning = (logvar_lower_bound is not None
                             and logvar_upper_bound is not None)
        if self._var_pinning:
            self._min_logvar = torch.nn.Parameter(
                torch.Tensor([logvar_lower_bound])
                * torch.ones(1, output_dim, dtype=torch.float32, requires_grad=True))
            self._max_logvar = torch.nn.Parameter(
                torch.Tensor([logvar_upper_bound])
                * torch.ones(1, output_dim, dtype=torch.float32, requires_grad=True))
        else:
            self._min_logvar = None
            self._max_logvar = None
        self._logvar_bound_loss_coef = logvar_bound_loss_coef
        
        # Set up metrics
        self._metrics = {
            'EV': SequentialExplainedVariance(),
            'IndvEV': SequentialExplainedVariance('raw_values'),
        }
        
        # Create mask tensor based on mask indices
        self._mask = torch.ones(input_dim).to(self.device)
        self._input_mask = False
        if len(mask_indices) > 0:
            self._mask[mask_indices] = 0
            self._input_mask = True
        
        # Loss configuration
        self.mse_wt = mse_wt
        self.nll_wt = nll_wt
        self.loss_fn_str = loss_fn_str
        
        # History tracking for single-step inference
        self._hidden_state = None
        self._history_buffer = None
        self._time_step = 0
        
        # Load pretrained weights if specified
        if load_dir is not None:
            path = os.path.join(load_dir, str(seed))
            checkpoint_path = None
            for root, dirs, files in os.walk(path):
                if 'checkpoints' in dirs:
                    checkpoint_path = os.path.join(root, 'checkpoints')
                    break
            if checkpoint_path is None:
                raise ValueError(f'Checkpoint directory not found in {path}')
            checkpoints = os.listdir(checkpoint_path)
            if not len(checkpoints):
                raise ValueError(f'No checkpoints found in {checkpoint_path}')
            epochs = [int(ck.split('-')[0].split('=')[1]) for ck in checkpoints]
            epidx = np.argmax(epochs)
            model_path = os.path.join(checkpoint_path, checkpoints[epidx])
            print("\n ---Loading model from ", model_path, "\n")
            
            checkpoint = torch.load(model_path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
        
        # Freezing logic
        self.freeze_all_but = kwargs.get('freeze_all_but', None)
        if self.freeze_all_but == "logvar_net":
            # Freeze all layers except for logvar_net layer
            for param in self._encoder.parameters():
                param.requires_grad = False

            if self._use_positional_encoding:
                self._positional_encoding.requires_grad = False
            
            for block in self._transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            
            if self._use_layer_norm:
                for param in self._layer_norm.parameters():
                    param.requires_grad = False
            
            for param in self._decoder.parameters():
                param.requires_grad = False
            
            # Set the logvar_net layer to be trainable
            for param in self._decoder._logvar_head.parameters():
                param.requires_grad = True
        
        elif self.freeze_all_but == "output_layer_block_4":
            # Freeze all layers except for decoder layer and last transformer block
            for param in self._encoder.parameters():
                param.requires_grad = False

            if self._use_positional_encoding:
                self._positional_encoding.requires_grad = False
            
            # Freeze all but last transformer block
            for i, block in enumerate(self._transformer_blocks):
                if i < len(self._transformer_blocks) - 1:
                    for param in block.parameters():
                        param.requires_grad = False
            
            if self._use_layer_norm:
                for param in self._layer_norm.parameters():
                    param.requires_grad = False
            
            # Set the decoder to be trainable
            for param in self._decoder._mean_head.parameters():
                param.requires_grad = True
            
            for param in self._decoder._logvar_head.parameters():
                param.requires_grad = True
            
            # Last transformer block remains trainable
        
        elif self.freeze_all_but == "output_layer":
            # Freeze all layers except for decoder output layer
            for param in self._encoder.parameters():
                param.requires_grad = False

            if self._use_positional_encoding:
                self._positional_encoding.requires_grad = False
            
            for block in self._transformer_blocks:
                for param in block.parameters():
                    param.requires_grad = False
            
            if self._use_layer_norm:
                for param in self._layer_norm.parameters():
                    param.requires_grad = False
            
            # Set the decoder last to be trainable
            for param in self._decoder._mean_head.parameters():
                param.requires_grad = True
            
            for param in self._decoder._logvar_head.parameters():
                param.requires_grad = True

    def get_net_out(self, batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Get the output of the network and organize into dictionary.

        Args:
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim)
                * y: (Batch_size, Sequence Length, dim)
                * mask: (Batch_size, Sequence Length, 1)

        Returns:
            Dictionary of name to tensor.
        """
        if self._input_mask:
            input_batch = batch[0] * self._mask.to(self.device)
        else:
            input_batch = batch[0]
        
        # Encode inputs
        encoded = self._encoder(input_batch)
        
        # Apply layer norm
        if self._use_layer_norm:
            encoded = self._layer_norm(encoded)
        
        # Add positional encoding
        if self._use_positional_encoding:
            seq_len = encoded.shape[1]
            encoded = encoded + self._positional_encoding[:, :seq_len, :]
        
        # Apply transformer blocks
        for transformer_block in self._transformer_blocks:
            encoded = transformer_block(encoded)
        
        # Decode to mean and logvar
        mean, logvar = self._decoder(encoded)
        
        return {'mean': mean, 'logvar': logvar}

    def loss(self, net_out: Dict[str, torch.Tensor], batch: Sequence[torch.Tensor]) -> \
            Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute the loss function.

        Args:
            net_out: The output of the network.
            batch: The batch passed into the network. This is expected to be a tuple
                * x: (Batch_size, Sequence Length, dim)
                * y: (Batch_size, Sequence Length, dim)
                * mask: (Batch_size, Sequence Length, 1)

        Returns:
            The loss and a dictionary of other statistics.
        """
        mean = net_out['mean']
        logvar = net_out['logvar']
        y, mask = batch[1:]
        mask[:, :self._warm_up_period, :] = 0
        sq_diffs = (mean * mask - y * mask).pow(2)
        mse = torch.mean(sq_diffs)
        
        if self.loss_fn_str == "NLL":
            loss = torch.mean(torch.exp(-logvar) * sq_diffs + logvar * mask)
        elif self.loss_fn_str == "MSE":
            loss = mse
        elif self.loss_fn_str == "NLL+MSE":
            loss = torch.mean(torch.exp(-logvar) * sq_diffs + logvar * mask) * self.nll_wt + mse * self.mse_wt
        else:
            raise ValueError(f'Cannot recognize loss function {self.loss_fn_str}, select from NLL, MSE, NLL+MSE')
        
        stats = dict()
        stats['mean/mean'] = (mean * mask).mean().item()
        stats['logvar/mean'] = (logvar * mask).mean().item()
        stats['nll/mean'] = torch.mean(torch.exp(-logvar) * sq_diffs).item()
        stats['mse/mean'] = torch.mean((mask * mse)).item()
        
        if self._var_pinning:
            bound_loss = self._logvar_bound_loss_coef * \
                         torch.abs(self._max_logvar - self._min_logvar).mean()
            stats['bound_loss'] = bound_loss.item()
            stats['logvar_lower_bound/mean'] = self._min_logvar.mean().item()
            stats['logvar_upper_bound/mean'] = self._max_logvar.mean().item()
            stats['logvar_bound_difference'] = (
                        self._max_logvar - self._min_logvar).mean().item()
            loss += bound_loss
        
        stats['loss'] = loss.item()
        return loss, stats

    def single_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
            torch.Tensor, Dict[str, Any]]:
        """Get the output for a single sample in the model.

        Args:
            net_in: The input for the network with expected shape (batch size, dim)

        Returns:
            The predictions for a single function sample.
        """
        # Initialize history buffer if needed
        if self._history_buffer is None:
            self._history_buffer = torch.zeros(
                net_in.shape[0], self._block_size, net_in.shape[1],
                device=self.device
            )
            self._time_step = 0
        else:
            if self._history_buffer.shape[0] != net_in.shape[0]:
                raise ValueError('Number of inputs does not match previously given '
                                 f'number. Expected {self._history_buffer.shape[0]} but received'
                                 f' {net_in.shape[0]}.')
        
        # Update history buffer
        if self._time_step >= self._block_size:
            # Shift buffer and add new input
            self._history_buffer = torch.cat([
                self._history_buffer[:, 1:, :],
                net_in.unsqueeze(1)
            ], dim=1)
        else:
            self._history_buffer[:, self._time_step, :] = net_in
        
        with torch.no_grad():
            if self._input_mask:
                input_batch = self._history_buffer * self._mask.to(self.device)
            else:
                input_batch = self._history_buffer
            
            # Encode
            encoded = self._encoder(input_batch)
            
            if self._use_layer_norm:
                encoded = self._layer_norm(encoded)
            
            # Add positional encoding
            if self._use_positional_encoding:
                seq_len = encoded.shape[1]
                encoded = encoded + self._positional_encoding[:, :seq_len, :]
            
            # Apply transformer blocks
            for transformer_block in self._transformer_blocks:
                encoded = transformer_block(encoded)
            
            # Decode
            mean_predictions, logvar_predictions = self._decoder(encoded)
            
            # Get prediction at current time step
            pred_idx = min(self._time_step, mean_predictions.shape[1] - 1)
            mean_predictions = mean_predictions[:, pred_idx, :]
            logvar_predictions = logvar_predictions[:, pred_idx, :]
        
        std_predictions = (0.5 * logvar_predictions).exp()
        
        if self._sample_mode == sampling_modes.SAMPLE_FROM_DIST:
            predictions = (torch.randn_like(mean_predictions) * std_predictions
                           + mean_predictions)
        else:
            predictions = mean_predictions
        
        info = {'predictions': predictions,
                'mean_predictions': mean_predictions,
                'std_predictions': std_predictions}
        
        if self._record_history:
            self._time_step += 1
        
        return predictions, info

    def multi_sample_output_from_torch(self, net_in: torch.Tensor) -> Tuple[
            torch.Tensor, Dict[str, Any]]:
        """Get the output where each input is assumed to be from a different sample.

        Args:
            net_in: The input for the network.

        Returns:
            The deltas for next states and dictionary of info.
        """
        return self.single_sample_output_from_torch(net_in)

    @property
    def metrics(self) -> Dict[str, Callable[[torch.Tensor], torch.Tensor]]:
        return self._metrics

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def weight_decay(self) -> float:
        return self._weight_decay

    @property
    def sample_mode(self) -> str:
        """The sample mode is the method that in which we get next state."""
        return self._sample_mode

    @sample_mode.setter
    def sample_mode(self, mode: str) -> None:
        """Set the sample mode to the appropriate mode."""
        if mode not in [sampling_modes.SAMPLE_FROM_DIST,
                        sampling_modes.RETURN_MEAN]:
            raise ValueError(
                f'TPNN sample mode must either be {sampling_modes.SAMPLE_FROM_DIST} '
                f'or {sampling_modes.RETURN_MEAN}, but received {mode}.')
        self._sample_mode = mode

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def record_history(self) -> bool:
        """Whether to keep track of the quantities being fed into the neural net."""
        return self._record_history

    @record_history.setter
    def record_history(self, mode: bool) -> None:
        """Set whether to keep track of quantities being fed into the neural net."""
        self._record_history = mode

    @property
    def warm_up_period(self) -> int:
        """Amount of data to take in before starting to predict"""
        return self._warm_up_period

    def clear_history(self) -> None:
        """Clear the history."""
        self._hidden_state = None
        self._history_buffer = None
        self._time_step = 0

    def reset(self) -> None:
        """Reset the dynamics model."""
        self.clear_history()
