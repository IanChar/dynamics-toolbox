"""
Encoder-GRU model that extracts only the encoder and GRU components from RPNN.
This model takes inputs and outputs the next hidden state only.

Author: Generated for RPNN state extraction
Date: February 2, 2026
"""
from typing import Dict, Tuple, Any, Optional

import os
import torch
import torch.nn as nn
import numpy as np


class RPNNEncoderGRU(nn.Module):
    """Encoder-GRU network that outputs hidden states only."""

    def __init__(
            self,
            input_dim: int,
            encode_dim: int,
            rnn_num_layers: int,
            rnn_hidden_size: int,
            rnn_type: str = 'gru',
            use_layer_norm: bool = True,
            mask_indices: Optional[list] = None,
            load_from_rpnn: Optional[str] = None,  # Path to trained RPNN checkpoint
            load_from_checkpoint: Optional[str] = None,  # Path to saved encoder-gru checkpoint
            seed: Optional[int] = None,
            device: str = 'cpu',
    ):
        """Constructor.

        Args:
            input_dim: The input dimension.
            encode_dim: The dimension of the encoder output.
            rnn_num_layers: Number of layers in the memory unit.
            rnn_hidden_size: The number hidden units in the memory unit.
            rnn_type: Name of the rnn type. Can accept GRU or LSTM.
            use_layer_norm: Whether to use layer norm.
            mask_indices: The indices to mask.
            load_from_rpnn: Path to trained RPNN model directory to load weights from.
            load_from_checkpoint: Path to saved RPNNEncoderGRU checkpoint (faster loading).
            seed: Random seed for loading specific model checkpoint (only for load_from_rpnn).
            device: Device to run the model on ('cpu' or 'cuda').
        """
        super().__init__()
        
        self.device = torch.device(device)
        
        # We'll load the encoder architecture from the checkpoint
        self._encoder = None
        self._use_layer_norm = use_layer_norm
        
        # Initialize GRU
        self.rnn_type = rnn_type.lower()
        if rnn_type.lower() == 'gru':
            rnn_class = torch.nn.GRU
        elif rnn_type.lower() == 'lstm':
            rnn_class = torch.nn.LSTM
        else:
            raise ValueError(f'Cannot recognize RNN type {rnn_type}')
        
        self._memory_unit = rnn_class(
            encode_dim, 
            rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True
        )
        self._memory_unit = self._memory_unit.to(self.device)
        
        # Store dimensions
        self._input_dim = input_dim
        self._output_dim = rnn_hidden_size
        self._encode_dim = encode_dim
        self._hidden_size = rnn_hidden_size
        self._num_layers = rnn_num_layers
        self._hidden_state = None
        self._record_history = True
        
        # Create mask tensor based on mask indices
        self._mask = torch.ones(input_dim).to(self.device)
        self._input_mask = False
        if mask_indices is not None and len(mask_indices) > 0:
            self._mask[mask_indices] = 0
            self._input_mask = True
        
        # Load weights from trained RPNN model or saved checkpoint
        if load_from_rpnn is not None and load_from_checkpoint is not None:
            raise ValueError(
                "Cannot specify both load_from_rpnn and load_from_checkpoint. "
                "Please provide only one."
            )
        elif load_from_rpnn is not None:
            self._load_weights_from_rpnn(load_from_rpnn, seed)
        elif load_from_checkpoint is not None:
            self._load_from_checkpoint(load_from_checkpoint)
        else:
            raise ValueError(
                "RPNNEncoderGRU must be initialized with either load_from_rpnn or "
                "load_from_checkpoint parameter. This model is designed to load "
                "weights from a trained model."
            )

    def _load_weights_from_rpnn(self, load_dir: str, seed: Optional[int] = None):
        """Load encoder and GRU weights from a trained RPNN model.
        
        Args:
            load_dir: Directory containing the trained RPNN model.
            seed: Seed to identify which checkpoint to load.
        """
        # Find checkpoint path
        if seed is not None:
            path = os.path.join(load_dir, str(seed))
        else:
            path = load_dir
            
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
        
        # Get the latest checkpoint
        epochs = [int(ck.split('-')[0].split('=')[1]) for ck in checkpoints]
        epidx = np.argmax(epochs)
        model_path = os.path.join(checkpoint_path, checkpoints[epidx])
        
        print(f"\n--- Loading Encoder-GRU weights from RPNN: {model_path} ---\n")
        
        # Load the checkpoint - extract only state_dict to avoid model instantiation issues
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint['state_dict']
        
        # Extract encoder, memory_unit, and layer_norm weights
        encoder_weights = {}
        memory_weights = {}
        layer_norm_weights = {}
        
        for key, value in state_dict.items():
            if key.startswith('_encoder.'):
                encoder_weights[key] = value
            elif key.startswith('_memory_unit.'):
                memory_weights[key] = value
            elif key.startswith('_layer_norm.') and self._use_layer_norm:
                layer_norm_weights[key] = value
        
        print(f"Extracted {len(encoder_weights)} encoder tensors")
        print(f"Extracted {len(memory_weights)} GRU tensors")
        print(f"Extracted {len(layer_norm_weights)} layer norm tensors")
        
        # Build encoder from loaded weights by reconstructing the architecture
        print("Reconstructing encoder from weights...")
        self._encoder = self._build_encoder_from_weights(encoder_weights, self._input_dim, self._encode_dim)
        
        # Build layer norm if needed
        if self._use_layer_norm and layer_norm_weights:
            self._layer_norm = nn.LayerNorm(self._encode_dim)
            # Load layer norm weights
            layer_norm_state = {k.replace('_layer_norm.', ''): v for k, v in layer_norm_weights.items()}
            self._layer_norm.load_state_dict(layer_norm_state)
        else:
            self._layer_norm = None
        
        # Load encoder weights
        encoder_state = {k.replace('_encoder.', ''): v for k, v in encoder_weights.items()}
        self._encoder.load_state_dict(encoder_state, strict=True)
        
        # Load GRU weights
        memory_state = {k.replace('_memory_unit.', ''): v for k, v in memory_weights.items()}
        self._memory_unit.load_state_dict(memory_state, strict=True)
        
        print(f"Successfully loaded and reconstructed model with {len(encoder_weights) + len(memory_weights) + len(layer_norm_weights)} parameter tensors")
        print()
    
    def _load_from_checkpoint(self, checkpoint_path: str):
        """Load from a saved RPNNEncoderGRU checkpoint.
        
        Args:
            checkpoint_path: Path to the saved checkpoint file.
        """
        print(f"\n--- Loading from saved checkpoint: {checkpoint_path} ---\n")
        
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Verify this is the right type of checkpoint
        if 'model_type' not in checkpoint or checkpoint['model_type'] != 'RPNNEncoderGRU':
            raise ValueError(
                f"Invalid checkpoint format. Expected RPNNEncoderGRU checkpoint, "
                f"but got: {checkpoint.get('model_type', 'unknown')}"
            )
        
        # Extract weights
        encoder_weights = checkpoint['encoder_weights']
        memory_weights = checkpoint['memory_weights']
        layer_norm_weights = checkpoint.get('layer_norm_weights', {})
        
        print(f"Loaded {len(encoder_weights)} encoder tensors")
        print(f"Loaded {len(memory_weights)} GRU tensors")
        print(f"Loaded {len(layer_norm_weights)} layer norm tensors")
        
        # Reconstruct encoder
        print("Reconstructing encoder from weights...")
        self._encoder = self._build_encoder_from_weights(encoder_weights, self._input_dim, self._encode_dim)
        
        # Build layer norm if needed
        if self._use_layer_norm and layer_norm_weights:
            self._layer_norm = nn.LayerNorm(self._encode_dim)
            layer_norm_state = {k.replace('_layer_norm.', ''): v for k, v in layer_norm_weights.items()}
            self._layer_norm.load_state_dict(layer_norm_state)
        else:
            self._layer_norm = None
        
        # Load weights
        encoder_state = {k.replace('_encoder.', ''): v for k, v in encoder_weights.items()}
        self._encoder.load_state_dict(encoder_state, strict=True)
        
        memory_state = {k.replace('_memory_unit.', ''): v for k, v in memory_weights.items()}
        self._memory_unit.load_state_dict(memory_state, strict=True)
        
        print(f"Successfully loaded model from checkpoint")
        print()
    
    def _build_encoder_from_weights(self, encoder_weights: dict, input_dim: int, output_dim: int) -> nn.Module:
        """Reconstruct encoder architecture from loaded weights.
        
        Args:
            encoder_weights: Dictionary of encoder weights from checkpoint
            input_dim: Input dimension
            output_dim: Output dimension
            
        Returns:
            Reconstructed encoder module
        """
        # Analyze weights to determine architecture
        # Check for _net.linear_X structure (from MLP)
        layer_indices = set()
        for key in encoder_weights.keys():
            if '_net.linear_' in key:
                # Extract layer index (e.g., "_encoder._net.linear_0.weight" -> 0)
                parts = key.split('_net.linear_')[-1].split('.')
                if parts[0].isdigit():
                    layer_indices.add(int(parts[0]))
        
        if not layer_indices:
            # Try _layers.X structure
            for key in encoder_weights.keys():
                if '_layers.' in key:
                    parts = key.split('_layers.')[-1].split('.')
                    if parts[0].isdigit():
                        layer_indices.add(int(parts[0]))
        
        if not layer_indices:
            raise ValueError("Could not determine encoder architecture from weights")
        
        max_layer_idx = max(layer_indices)
        print(f"Detected MLP encoder with {max_layer_idx + 1} layers")
        
        # Build the MLP
        layers = nn.ModuleDict()
        for i in range(max_layer_idx + 1):
            # Try _net.linear_X first
            weight_key = f'_encoder._net.linear_{i}.weight'
            if weight_key not in encoder_weights:
                # Try _layers.X
                weight_key = f'_encoder._layers.{i}.weight'
            
            if weight_key in encoder_weights:
                out_features, in_features = encoder_weights[weight_key].shape
                layers[str(i)] = nn.Linear(in_features, out_features)
        
        # Create a simple sequential wrapper that matches the structure
        class EncoderModule(nn.Module):
            def __init__(self, layers_dict):
                super().__init__()
                self._net = nn.ModuleDict()
                for i, (key, layer) in enumerate(layers_dict.items()):
                    self._net[f'linear_{i}'] = layer
                
            def forward(self, x):
                for i in range(len(self._net)):
                    x = self._net[f'linear_{i}'](x)
                    # Apply ReLU to all but last layer
                    if i < len(self._net) - 1:
                        x = torch.relu(x)
                return x
        
        return EncoderModule(layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and GRU.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Hidden states of shape (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        """
        # Handle single time step input
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
            squeeze_output = True
        
        if self._input_mask:
            x = x * self._mask
        
        # Encode
        encoded = self._encoder(x)
        if self._use_layer_norm and self._layer_norm is not None:
            encoded = self._layer_norm(encoded)
        
        # Pass through GRU
        mem_out = self._memory_unit(encoded)[0]
        
        # Remove sequence dimension if input was 2D
        if squeeze_output:
            mem_out = mem_out.squeeze(1)
        
        return mem_out

    def get_hidden_state(
        self, 
        net_in: torch.Tensor,
        return_history: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get the hidden state output while maintaining internal state across calls.

        Args:
            net_in: The input for the network with shape (batch size, dim)
            return_history: Whether to return the internal hidden state history

        Returns:
            The hidden state output and info dictionary.
        """
        if self._hidden_state is None:
            if self.rnn_type == 'gru':
                self._hidden_state = torch.zeros(
                    self._num_layers, 
                    net_in.shape[0],
                    self._hidden_size, 
                    device=self.device
                )
            else:
                self._hidden_state = tuple(
                    torch.zeros(
                        self._num_layers, 
                        net_in.shape[0],
                        self._hidden_size, 
                        device=self.device
                    )
                    for _ in range(2)
                )
        else:
            tocompare = (self._hidden_state if self.rnn_type == 'gru'
                         else self._hidden_state[0])
            if tocompare.shape[1] != net_in.shape[0]:
                raise ValueError(
                    f'Number of inputs does not match previously given number. '
                    f'Expected {tocompare.shape[1]} but received {net_in.shape[0]}.'
                )
        
        with torch.no_grad():
            if self._input_mask:
                net_in = net_in * self._mask
            encoded = self._encoder(net_in).unsqueeze(1)
            if self._use_layer_norm and self._layer_norm is not None:
                encoded = self._layer_norm(encoded)
            mem_out, hidden_out = self._memory_unit(encoded, self._hidden_state)
            
            if self._record_history:
                self._hidden_state = hidden_out
            
            # Squeeze to remove sequence dimension
            hidden_state_output = mem_out.squeeze(1)
        
        info = {'hidden_state_output': hidden_state_output}
        if return_history:
            info['hidden_state'] = self._hidden_state
            
        return hidden_state_output, info

    def clear_history(self) -> None:
        """Clear the internal hidden state."""
        self._hidden_state = None

    def reset(self) -> None:
        """Reset the model (alias for clear_history)."""
        self.clear_history()

    @property
    def record_history(self) -> bool:
        """Whether to keep track of hidden states across calls."""
        return self._record_history

    @record_history.setter
    def record_history(self, mode: bool) -> None:
        """Set whether to keep track of hidden states across calls."""
        self._record_history = mode

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim
    
    def predict(self, input_data, hidden_state=None):
        """Predict method expected by fusion_env.
        
        Args:
            input_data: numpy array of shape (batch_size, input_dim)
            hidden_state: numpy array of shape (batch_size, hidden_dim) or None
            
        Returns:
            output: Not used (dummy return)
            info_dict: Dictionary containing 'hidden_state' with the next hidden state
        """
        import numpy as np
        
        # Convert input to torch tensor
        if isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
        else:
            input_tensor = input_data
            
        # Set the internal hidden state if provided
        if hidden_state is not None:
            if isinstance(hidden_state, np.ndarray):
                hidden_state_tensor = torch.from_numpy(hidden_state).float().to(self.device)
            else:
                hidden_state_tensor = hidden_state
            
            # Convert (batch_size, hidden_dim) to (num_layers, batch_size, hidden_dim)
            if hidden_state_tensor.dim() == 2:
                hidden_state_tensor = hidden_state_tensor.unsqueeze(0).repeat(
                    self._num_layers, 1, 1
                )
            
            self._hidden_state = hidden_state_tensor
        
        # Get the next hidden state
        output, info = self.get_hidden_state(input_tensor, return_history=True)
        
        # Extract the hidden state from the internal state (last layer)
        if 'hidden_state' in info:
            # The internal state is (num_layers, batch_size, hidden_dim)
            # We return the last layer as a torch tensor
            info_dict = {'hidden_state': info['hidden_state'][-1]}  # Keep as torch tensor
        else:
            # Fallback to the output
            info_dict = {'hidden_state': output}
        
        return output, info_dict
    
    @property
    def hidden_state_dim(self):
        """Property expected by fusion_env for latent dimension."""
        return self._output_dim
    
    def to(self, device):
        """Move model to device and update internal device tracking."""
        # Convert device string to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
        
        self.device = device
        
        # Move all submodules
        if self._encoder is not None:
            self._encoder = self._encoder.to(device)
        if self._memory_unit is not None:
            self._memory_unit = self._memory_unit.to(device)
        if self._layer_norm is not None:
            self._layer_norm = self._layer_norm.to(device)
        if self._mask is not None:
            self._mask = self._mask.to(device)
        
        # Move hidden state if it exists
        if self._hidden_state is not None:
            if isinstance(self._hidden_state, tuple):
                self._hidden_state = tuple(h.to(device) for h in self._hidden_state)
            else:
                self._hidden_state = self._hidden_state.to(device)
        
        # Call parent to() for any remaining parameters
        super().to(device)
        
        return self
    
    def save_checkpoint(self, save_dir: str, source_rpnn_path: Optional[str] = None):
        """Save the model weights and config to a directory.
        
        This saves:
        1. weights.pt - Model weights (encoder, GRU, layer norm)
        2. config.yaml - Model configuration for easy loading
        
        Args:
            save_dir: Directory where the checkpoint should be saved.
            source_rpnn_path: Optional path to the original RPNN model used for extraction.
        """
        import yaml
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect all weights
        encoder_weights = {f'_encoder.{k}': v for k, v in self._encoder.state_dict().items()}
        memory_weights = {f'_memory_unit.{k}': v for k, v in self._memory_unit.state_dict().items()}
        
        layer_norm_weights = {}
        if self._layer_norm is not None:
            layer_norm_weights = {f'_layer_norm.{k}': v for k, v in self._layer_norm.state_dict().items()}
        
        # Create checkpoint with weights
        checkpoint = {
            'model_type': 'RPNNEncoderGRU',
            'encoder_weights': encoder_weights,
            'memory_weights': memory_weights,
            'layer_norm_weights': layer_norm_weights,
        }
        
        # Save weights
        weights_path = os.path.join(save_dir, 'weights.pt')
        torch.save(checkpoint, weights_path)
        
        # Create config
        config = {
            'model_type': 'RPNNEncoderGRU',
            'input_dim': self._input_dim,
            'encode_dim': self._encode_dim,
            'rnn_num_layers': self._num_layers,
            'rnn_hidden_size': self._hidden_size,
            'rnn_type': self.rnn_type,
            'use_layer_norm': self._use_layer_norm,
            'source_rpnn_path': source_rpnn_path,
        }
        
        # Save config
        config_path = os.path.join(save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"\n✓ Saved RPNNEncoderGRU checkpoint to: {save_dir}")
        print(f"  - weights.pt: {len(encoder_weights) + len(memory_weights) + len(layer_norm_weights)} tensors")
        print(f"  - config.yaml: model configuration")
        if source_rpnn_path:
            print(f"  - source RPNN: {source_rpnn_path}")
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_dir: str, device: str = 'cpu'):
        """Load model from a saved checkpoint directory (simplified loading).
        
        This is the recommended way to load a saved model. Simply provide the
        directory path and the model will be loaded with the saved configuration.
        
        Args:
            checkpoint_dir: Directory containing weights.pt and config.yaml
            device: Device to load the model on ('cpu' or 'cuda')
            
        Returns:
            Loaded RPNNEncoderGRU model
            
        Example:
            model = RPNNEncoderGRU.load_from_checkpoint('/path/to/checkpoint')
        """
        import yaml
        
        config_path = os.path.join(checkpoint_dir, 'config.yaml')
        weights_path = os.path.join(checkpoint_dir, 'weights.pt')
        
        if not os.path.exists(config_path):
            raise ValueError(f"Config file not found: {config_path}")
        if not os.path.exists(weights_path):
            raise ValueError(f"Weights file not found: {weights_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"\n--- Loading from checkpoint: {checkpoint_dir} ---\n")
        print(f"Configuration:")
        for key, value in config.items():
            if key != 'source_rpnn_path':
                print(f"  {key}: {value}")
        if config.get('source_rpnn_path'):
            print(f"  source_rpnn_path: {config['source_rpnn_path']}")
        print()
        
        # Create model with loaded config
        model = cls(
            input_dim=config['input_dim'],
            encode_dim=config['encode_dim'],
            rnn_num_layers=config['rnn_num_layers'],
            rnn_hidden_size=config['rnn_hidden_size'],
            rnn_type=config.get('rnn_type', 'gru'),
            use_layer_norm=config.get('use_layer_norm', True),
            load_from_checkpoint=weights_path,
            device=device,
        )
        
        return model


if __name__ == "__main__":
    """
    Example: Load encoder and GRU weights from a trained RPNN model,
    then save to a standalone checkpoint for faster loading in the future.
    """
    print("="*70)
    print("STEP 1: Loading RPNNEncoderGRU from trained RPNN checkpoint")
    print("="*70)
    
    # Path to your trained RPNN checkpoint directory
    rpnn_checkpoint_dir = '/home/scratch/rsonker/dynamics_models/rpnn_noshape_gas_bms_step_one_mse_1f_red_v2'
    
    # Create model and load weights from trained RPNN (slow, extracts from full RPNN)

    model = RPNNEncoderGRU(
        input_dim=23,  # Must match RPNN input_dim (state + action)
        encode_dim=512,  # Must match RPNN encode_dim
        rnn_num_layers=1,  # Must match RPNN rnn_num_layers
        rnn_hidden_size=32,  # Must match RPNN rnn_hidden_size
        rnn_type='gru',
        use_layer_norm=True,  # Must match RPNN use_layer_norm
        load_from_rpnn=rpnn_checkpoint_dir,
        seed=0,
        device='cpu',
    )
    
    print("\n" + "="*70)
    print("STEP 2: Testing forward pass")
    print("="*70)
    
    # Test forward pass
    batch_size = 4
    sample_input = torch.randn(batch_size, 23)
    hidden_state, info = model.get_hidden_state(sample_input)
    
    print(f"\nInput shape: {sample_input.shape}")
    print(f"Output (hidden state) shape: {hidden_state.shape}")
    
    # Test stateless forward
    output = model.forward(sample_input)
    print(f"Forward output shape: {output.shape}")
    
    print("\n" + "="*70)
    print("STEP 3: Saving converted weights with config for faster loading")
    print("="*70)
    
    # Save the extracted weights to a standalone checkpoint with config
    save_dir = '/home/scratch/rsonker/dynamics_models/rpnn_observer_1f_red_v2'
    model.save_checkpoint(
        save_dir=save_dir,
        source_rpnn_path=rpnn_checkpoint_dir  # Store original RPNN path
    )
    
    print("\n" + "="*70)
    print("STEP 4: Loading from saved checkpoint (FAST & SIMPLE!)")
    print("="*70)
    
    # Now load directly from the saved checkpoint (much faster!)
    # Just provide the directory - config is loaded automatically!
    model_fast = RPNNEncoderGRU.load_from_checkpoint(
        checkpoint_dir=save_dir,
        device='cpu',
    )
    
    # Verify it works
    output_fast = model_fast.forward(sample_input)
    print(f"\nFast-loaded model output shape: {output_fast.shape}")
    
    # Verify outputs match
    max_diff = torch.abs(output - output_fast).max().item()
    print(f"Max difference between models: {max_diff:.2e}")
    
    if max_diff < 1e-6:
        print("\n✓ SUCCESS! Both loading methods produce identical results.")
        print(f"\nNext time, simply use:")
        print(f"  model = RPNNEncoderGRU.load_from_checkpoint('{save_dir}')")
    else:
        print(f"\n⚠ Warning: Models differ by {max_diff}")