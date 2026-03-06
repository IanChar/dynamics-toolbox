# RPNN Encoder-GRU Extraction

This module provides functionality to extract encoder and GRU components from a trained RPNN model and create a standalone model that only outputs hidden states.

## Overview

The `RPNNEncoderGRU` class is a lightweight model that contains only:
- Encoder network (from RPNN)
- GRU memory unit (from RPNN)
- Optional layer normalization (from RPNN)

Given inputs, it outputs the GRU hidden states instead of the full RPNN prediction (mean and variance).

**File:** `dynamics_toolbox/models/pl_models/sequential_models/rpnn_encoder_gru.py`

## Usage

### Recommended: Extract Once, Load Fast

**Step 1: Extract from RPNN and save (one-time operation):**

```python
from dynamics_toolbox.models.pl_models.sequential_models import RPNNEncoderGRU
import torch

# Load from RPNN (slower, but only do this once)
model = RPNNEncoderGRU(
    input_dim=55,
    encode_dim=512,
    rnn_num_layers=1,
    rnn_hidden_size=256,
    rnn_type='gru',
    use_layer_norm=True,
    load_from_rpnn='/path/to/rpnn/checkpoints',
    seed=0,
    device='cpu',
)

# Save extracted weights for future use
model.save_checkpoint('rpnn_encoder_gru.pt')
```

**Step 2: Load from saved checkpoint (fast, use this going forward):**

```python
# Load directly from saved checkpoint (instant!)
model = RPNNEncoderGRU(
    input_dim=55,
    encode_dim=512,
    rnn_num_layers=1,
    rnn_hidden_size=256,
    rnn_type='gru',
    use_layer_norm=True,
    load_from_checkpoint='rpnn_encoder_gru.pt',  # Fast!
    device='cpu',
)

# Use the model
input_data = torch.randn(4, 55)
hidden_state, info = model.get_hidden_state(input_data)
print(hidden_state.shape)  # (4, 256)
```

The model will automatically:
- Find the checkpoint in the specified directory
- Extract encoder, GRU, and layer_norm weights
- Verify all dimensions match
- Load the weights into the model

## Advanced Features

### Hidden State Management

The model maintains hidden states across sequential predictions:

```python
model.clear_history()  # Reset hidden states

# Sequential processing
for t in range(sequence_length):
    input_t = data[t]  # Shape: (batch_size, input_dim)
    hidden_t, info = model.single_sample_output_from_torch(input_t)
    # hidden_t shape: (batch_size, hidden_dim)
    # Hidden state is automatically maintained internally

model.clear_history()  # Clear when done
```

### Controlling History Recording

```python
# Disable history recording (hidden state won't update)
model.record_history = False
output, info = model.single_sample_output_from_torch(input_data)

# Re-enable history recording
model.record_history = True
```

## Model Architecture

```
Input (state + action)
    ↓
[Encoder Network] → encode_dim
    ↓
[Optional Layer Norm]
    ↓
[GRU] → hidden_size
    ↓
Hidden State Output
```

## Configuration Notes

### Important: Match RPNN Configuration

When creating `RPNNEncoderGRU`, ensure all dimensions match your trained RPNN:

- `input_dim`: Must match RPNN input dimension
- `encode_dim`: Must match RPNN encoder output dimension  
- `rnn_hidden_size`: Must match RPNN GRU hidden size
- `rnn_num_layers`: Must match RPNN GRU number of layers
- `encoder_cfg`: Should replicate RPNN encoder architecture
- `use_layer_norm`: Must match RPNN layer norm setting

### Finding RPNN Configuration

Check your RPNN training config file or hydra outputs to find these values.

## What Happens During Loading

When you set `load_from_rpnn='/path/to/rpnn/checkpoints'`, the model automatically:

1. **Finds the checkpoint**: Searches for the checkpoint directory (optionally using seed)
2. **Loads the latest epoch**: Selects the checkpoint with the highest epoch number
3. **Extracts relevant weights**: Pulls only encoder, GRU, and layer_norm parameters
4. **Verifies dimensions**: 
   - Checks encoder output matches GRU input
   - Validates all layer shapes match expected dimensions
   - Raises clear errors if mismatches are detected
5. **Loads weights**: Applies the extracted weights to the model

Example output during loading:
```
--- Loading Encoder-GRU weights from RPNN: /path/to/checkpoint.ckpt ---

Verifying layer dimensions...
All dimensions verified successfully!
Successfully loaded 38 parameter tensors.
```

## Troubleshooting

### Dimension Mismatch Errors

If you see dimension mismatch errors:

1. Check that `encode_dim` matches the encoder output in your RPNN config
2. Check that `rnn_hidden_size` matches the GRU hidden size
3. Verify `rnn_num_layers` matches the number of GRU layers
4. Use `verify_weight_compatibility()` to diagnose

### Checkpoint Not Found

Ensure the path points to a directory containing:
```
checkpoint_dir/
  └── seed_number/  (if using seed)
      └── lightning_logs/ or similar
          └── checkpoints/
              └── epoch=X-step=Y.ckpt
```

### Wrong Output Shape

The output shape should be `(batch_size, rnn_hidden_size)`. If it's wrong:
- Verify `output_dim` parameter equals `rnn_hidden_size`
- Check that inputs have correct shape `(batch_size, input_dim)`

## API Reference

### RPNNEncoderGRU

```python
class RPNNEncoderGRU(AbstractSequentialModel):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,  # Should equal rnn_hidden_size
        encode_dim: int,
        rnn_num_layers: int,
        rnn_hidden_size: int,
        encoder_cfg: DictConfig,
        rnn_type: str = 'gru',
        warm_up_period: int = 0,
        learning_rate: float = 1e-3,
        weight_decay: Optional[float] = 0.0,
        use_layer_norm: bool = True,
        mask_indices: Optional[Sequence[int]] = [],
        load_from_rpnn: Optional[str] = None,  # Path to RPNN checkpoint dir
        seed: Optional[int] = None,
        **kwargs,
    )
```

**Key Methods:**
- `single_sample_output_from_torch(net_in)`: Returns hidden state for input
- `clear_history()`: Resets the internal hidden state
- `reset()`: Alias for clear_history()

## Integration with Existing Code

The `RPNNEncoderGRU` inherits from `AbstractSequentialModel`, so it's compatible with existing dynamics toolbox infrastructure:

```python
# Use with model wrappers
from dynamics_toolbox.env_wrappers import ModelEnv

model_env = ModelEnv(
    model=encoder_gru_model,
    env=base_env,
    # ... other params
)
```

## License

Same as parent project.
