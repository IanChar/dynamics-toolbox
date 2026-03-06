# TPNN Implementation Summary

## What Was Created

I've created a **Transformer Probabilistic Neural Network (TPNN)** that serves as a drop-in replacement for RPNN in your dynamics-toolbox repository.

### Files Created

1. **`dynamics_toolbox/models/pl_models/sequential_models/tpnn.py`**
   - Main TPNN implementation (550+ lines)
   - Complete transformer-based architecture with self-attention
   - Maintains exact same interface as RPNN

2. **`example_configs/model/tpnn.yaml`**
   - Example configuration file for TPNN
   - Shows all configurable parameters
   - Provides sensible defaults

3. **`dynamics_toolbox/models/pl_models/sequential_models/TPNN_README.md`**
   - Comprehensive documentation
   - Usage examples and comparisons
   - Tips for best performance

4. **`scripts/compare_rpnn_tpnn.py`**
   - Comparison and testing script
   - Demonstrates interface equivalence
   - Shows sequential inference

5. **`dynamics_toolbox/models/pl_models/sequential_models/__init__.py`**
   - Updated to export TPNN

## Key Features Preserved from RPNN

✅ **All class methods and properties:**
- `get_net_out(batch)` - Forward pass on batch data
- `loss(net_out, batch)` - Loss computation with statistics
- `single_sample_output_from_torch(net_in)` - Single-step prediction
- `multi_sample_output_from_torch(net_in)` - Multi-sample prediction
- `clear_history()` - Reset history buffer
- `reset()` - Full reset

✅ **All properties:**
- `input_dim`, `output_dim`
- `learning_rate`, `weight_decay`
- `sample_mode`, `warm_up_period`
- `record_history`, `metrics`

✅ **All functionality:**
- Gaussian distribution predictions (mean, logvar)
- Variance pinning with bounds
- Input masking support
- Loss function options: NLL, MSE, NLL+MSE
- Pretrained weight loading
- Selective layer freezing (fine-tuning)
- History management for sequential inference

## Architecture Differences

| Component | RPNN | TPNN |
|-----------|------|------|
| **Sequence Modeling** | RNN/GRU/LSTM | Transformer (Multi-head Self-Attention) |
| **Memory** | Hidden state vector | Attention over full history buffer |
| **Processing** | Sequential | Parallel (during training) |
| **Context Window** | Implicit via hidden state | Explicit via block_size |
| **Long-range Dependencies** | Exponential decay | Direct attention |
| **Position Awareness** | Implicit in RNN | Learnable positional encoding |

## How to Use

### Simple Replacement

**Before (RPNN):**
```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.rpnn.RPNN
  input_dim: 10
  output_dim: 5
  encode_dim: 128
  rnn_num_layers: 2
  rnn_hidden_size: 256
  encoder_cfg: ...
  pnn_decoder_cfg: ...
```

**After (TPNN):**
```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.tpnn.TPNN
  input_dim: 10
  output_dim: 5
  encode_dim: 128  # Must be divisible by num_heads
  num_transformer_blocks: 4
  num_heads: 8
  block_size: 100
  encoder_cfg: ...
  pnn_decoder_cfg: ...
```

### Configuration Mapping

| RPNN Parameter | TPNN Equivalent | Notes |
|----------------|-----------------|-------|
| `rnn_num_layers` | `num_transformer_blocks` | Similar depth concept |
| `rnn_hidden_size` | `encode_dim` | Size of representations |
| `rnn_type` | N/A | TPNN uses self-attention |
| N/A | `num_heads` | New: number of attention heads |
| N/A | `block_size` | New: max sequence length |
| N/A | `dropout` | New: regularization |
| N/A | `use_positional_encoding` | New: position awareness |

## Implementation Details

### Transformer Architecture

Each transformer block contains:
1. **Layer Normalization**
2. **Multi-head Causal Self-Attention**
   - Causal masking (only attend to past)
   - Scaled dot-product attention
   - Multiple attention heads
3. **Residual Connection**
4. **Layer Normalization**
5. **Feed-Forward Network** (4x expansion)
6. **Residual Connection**

### Single-Step Inference

For sequential prediction (e.g., in RL environments):
```python
model = TPNN(...)
model.clear_history()  # Start new sequence

for observation in observations:
    prediction, info = model.single_sample_output_from_torch(observation)
    # info contains: predictions, mean_predictions, std_predictions
```

The model maintains a history buffer of size `block_size` and uses the transformer to attend over the full history when making predictions.

### Batch Training

For batch training on sequences:
```python
# batch = [x, y, mask]
# x: (batch_size, seq_length, input_dim)
# y: (batch_size, seq_length, output_dim)
# mask: (batch_size, seq_length, 1)

net_out = model.get_net_out(batch)  # Returns {mean, logvar}
loss, stats = model.loss(net_out, batch)
```

## Advantages of TPNN over RPNN

1. **Parallelization**: Training is faster on GPUs (sequences processed in parallel)
2. **Long-range Dependencies**: Direct attention to any past timestep
3. **Interpretability**: Attention weights show what model focuses on
4. **Scalability**: Performance improves with more data and larger models
5. **No Vanishing Gradients**: Direct connections via attention

## Considerations

1. **Memory**: O(n²) complexity in sequence length (vs O(n) for RNN)
2. **Block Size**: Must set `block_size` based on expected sequence lengths
3. **Hyperparameters**: More hyperparameters to tune (num_heads, num_blocks, dropout)
4. **Inference**: Slightly more complex than RNN hidden state

## Testing

Run the comparison script to verify equivalence:
```bash
cd /zfsauton2/home/rsonker/dynamics-toolbox
python scripts/compare_rpnn_tpnn.py
```

This will:
- Verify interface equivalence
- Test sequential inference
- Compare architectures
- Ensure drop-in compatibility

## Integration with Existing Pipelines

TPNN works with all existing:
- Data modules (classification, forward dynamics, etc.)
- Training scripts (`train.py`, `train_cb.py`)
- Evaluation pipelines
- Model storage/loading utilities
- Environment wrappers

Simply change the model configuration and everything else remains the same!

## Recommended Settings

For best results with TPNN:

```yaml
# Good starting point
encode_dim: 128  # Must be divisible by num_heads
num_transformer_blocks: 4-6
num_heads: 8
block_size: 100-200  # Based on your sequence lengths
dropout: 0.1-0.2
learning_rate: 1e-4  # Lower than typical RNN learning rates
use_layer_norm: true
use_positional_encoding: true
```

## Future Enhancements (Optional)

Potential improvements you could add:
1. **Bidirectional attention** for offline learning
2. **Flash attention** for longer sequences
3. **Relative positional encodings**
4. **Sparse attention patterns**
5. **Knowledge distillation from RPNN**

## Questions?

See `TPNN_README.md` for detailed documentation and examples.
