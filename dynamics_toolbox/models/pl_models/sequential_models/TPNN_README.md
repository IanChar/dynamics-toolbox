# TPNN: Transformer Probabilistic Neural Network

## Overview

TPNN (Transformer Probabilistic Neural Network) is a transformer-based sequential model that serves as a drop-in replacement for RPNN. Instead of using RNN/GRU/LSTM for sequence modeling, TPNN uses multi-head self-attention mechanisms (transformer blocks) to capture temporal dependencies.

## Key Features

- **Transformer Architecture**: Uses self-attention mechanisms for sequence modeling
- **Probabilistic Predictions**: Outputs Gaussian distributions (mean and log variance)
- **Drop-in Replacement**: Maintains the same interface as RPNN for seamless integration
- **Flexible Configuration**: Supports various loss functions, variance bounds, and fine-tuning options
- **Positional Encoding**: Optional learnable positional encodings for sequence awareness

## Architecture

```
Input Sequence
    ↓
Encoder (MLP)
    ↓
Layer Norm (optional)
    ↓
Positional Encoding (optional)
    ↓
Transformer Blocks (Multi-head Self-Attention + FFN)
    ↓
Decoder (PNN with mean and logvar heads)
    ↓
Output: (mean, logvar)
```

## Comparison with RPNN

| Feature | RPNN | TPNN |
|---------|------|------|
| Sequence Modeling | RNN/GRU/LSTM | Transformer (Self-Attention) |
| Memory Mechanism | Hidden state | Attention over full sequence |
| Parallelization | Sequential | Parallel (during training) |
| Long-range Dependencies | Limited by hidden state | Full sequence context |
| Positional Information | Implicit in RNN | Explicit positional encoding |

## Usage

### Basic Configuration

```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.tpnn.TPNN
  input_dim: 10
  output_dim: 5
  encode_dim: 128  # Must be divisible by num_heads
  num_transformer_blocks: 4
  num_heads: 8
  block_size: 100  # Maximum sequence length
  dropout: 0.1
  learning_rate: 1e-4
```

### Key Parameters

- **encode_dim**: Dimension of the encoded representation (must be divisible by num_heads)
- **num_transformer_blocks**: Number of transformer blocks to stack
- **num_heads**: Number of attention heads in each transformer block
- **block_size**: Maximum sequence length the model can handle
- **dropout**: Dropout probability for regularization
- **use_positional_encoding**: Whether to add learnable positional encodings

### Loss Functions

TPNN supports three loss function options via `loss_fn_str`:

1. **"NLL"** (Default): Negative log-likelihood loss
   - Loss = E[exp(-logvar) * (y - mean)² + logvar]
   
2. **"MSE"**: Mean squared error loss
   - Loss = E[(y - mean)²]
   
3. **"NLL+MSE"**: Combined loss
   - Loss = nll_wt * NLL + mse_wt * MSE

### Fine-tuning Options

TPNN supports selective layer freezing via `freeze_all_but`:

- **null** (default): All layers trainable
- **"logvar_net"**: Only logvar head trainable
- **"output_layer"**: Only decoder heads trainable
- **"output_layer_block_4"**: Decoder heads + last transformer block trainable

### Loading Pretrained Weights

```yaml
model:
  # ... other config ...
  load_dir: /path/to/checkpoint/dir
  seed: 42
```

## Example: Replacing RPNN with TPNN

### Original RPNN Config
```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.rpnn.RPNN
  input_dim: 10
  output_dim: 5
  encode_dim: 128
  rnn_num_layers: 2
  rnn_hidden_size: 256
  rnn_type: gru
```

### Equivalent TPNN Config
```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.tpnn.TPNN
  input_dim: 10
  output_dim: 5
  encode_dim: 128  # Same as RPNN
  num_transformer_blocks: 4  # Similar to rnn_num_layers
  num_heads: 8
  block_size: 100
```

## Implementation Details

### Transformer Blocks

Each transformer block consists of:
1. Layer normalization
2. Multi-head self-attention with causal masking
3. Residual connection
4. Layer normalization
5. Feed-forward network (4x expansion)
6. Residual connection

### Single-Step Inference

For sequential prediction (e.g., in RL environments):
- Maintains a history buffer of size `block_size`
- Adds new observations to the buffer
- Runs transformer over the full history
- Returns prediction for the current timestep

### Memory Management

- History buffer is automatically managed
- Call `clear_history()` or `reset()` to start a new sequence
- Set `record_history = False` to prevent buffer updates

## Advantages

1. **Parallelization**: Transformer blocks can process sequences in parallel during training
2. **Long-range Dependencies**: Self-attention captures dependencies across the entire sequence
3. **Interpretability**: Attention weights can be analyzed to understand what the model focuses on
4. **Scalability**: Performance typically improves with more data and model size

## Considerations

1. **Memory**: Transformers have O(n²) memory complexity in sequence length
2. **Sequence Length**: Limited by `block_size` parameter
3. **Computational Cost**: Self-attention can be more expensive than RNNs for very long sequences
4. **Hyperparameter Sensitivity**: Requires careful tuning of num_heads, num_blocks, etc.

## Tips for Best Performance

1. **Encode Dimension**: Should be divisible by num_heads for even distribution
2. **Block Size**: Set based on expected sequence lengths in your data
3. **Number of Heads**: Start with 4-8 heads; more heads = more expressive but more parameters
4. **Dropout**: Use 0.1-0.2 for regularization
5. **Learning Rate**: Transformers often work well with smaller learning rates (1e-4 to 1e-3)
6. **Positional Encoding**: Generally helpful for sequence-dependent tasks

## Citation

If you use TPNN in your research, please cite the original RPNN work and note that TPNN is a transformer-based variant.
