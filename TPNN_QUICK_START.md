# TPNN Quick Start Guide

## 1. Minimal Example

```python
from dynamics_toolbox.models.pl_models.sequential_models.tpnn import TPNN
from omegaconf import DictConfig
import torch

# Create model
model = TPNN(
    input_dim=10,
    output_dim=5,
    encode_dim=128,
    num_transformer_blocks=4,
    num_heads=8,
    block_size=100,
    encoder_cfg=DictConfig({
        '_target_': 'dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP',
        'hidden_sizes': [256, 256],
        'hidden_activation': 'torch.nn.functional.relu',
        'output_activation': 'torch.nn.functional.relu',
    }),
    pnn_decoder_cfg=DictConfig({
        '_target_': 'dynamics_toolbox.models.pl_models.pnn.PNN',
        'encoder_output_dim': 128,
        'encoder_cfg': DictConfig({
            '_target_': 'dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP',
            'hidden_sizes': [256],
            'hidden_activation': 'torch.nn.functional.relu',
            'output_activation': 'torch.nn.functional.relu',
        }),
        'mean_net_cfg': DictConfig({
            '_target_': 'dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP',
            'hidden_sizes': [],
            'output_activation': None,
        }),
        'logvar_net_cfg': DictConfig({
            '_target_': 'dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP',
            'hidden_sizes': [],
            'output_activation': None,
        }),
    }),
)

# Batch training
x = torch.randn(4, 20, 10)  # (batch, seq_len, input_dim)
y = torch.randn(4, 20, 5)   # (batch, seq_len, output_dim)
mask = torch.ones(4, 20, 1) # (batch, seq_len, 1)

net_out = model.get_net_out([x, y, mask])
loss, stats = model.loss(net_out, [x, y, mask])

print(f"Loss: {loss.item()}")
print(f"Statistics: {stats}")
```

## 2. Sequential Inference (RL/Online)

```python
# Initialize
model.clear_history()

# Run sequential predictions
for t in range(num_steps):
    observation = get_observation()  # Shape: (batch_size, input_dim)
    prediction, info = model.single_sample_output_from_torch(observation)
    
    # info contains:
    # - predictions: sampled prediction
    # - mean_predictions: mean of distribution
    # - std_predictions: std of distribution

# Reset for new episode
model.reset()
```

## 3. Using in Hydra Config

Create `configs/experiment/my_tpnn_experiment.yaml`:

```yaml
# @package _global_

defaults:
  - /model: tpnn
  - /data_module: forward_dynamics_data_module
  - /trainer: pl_trainer

model:
  input_dim: ${data_module.input_dim}
  output_dim: ${data_module.output_dim}
  encode_dim: 128
  num_transformer_blocks: 4
  num_heads: 8
  block_size: 100
  learning_rate: 1e-4
  loss_fn_str: "NLL"

data_module:
  batch_size: 64
  data_path: data/pendulum_100k.hdf5

trainer:
  max_epochs: 100
```

Run with:
```bash
python train.py experiment=my_tpnn_experiment
```

## 4. Converting RPNN Config to TPNN

### Original RPNN Config
```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.rpnn.RPNN
  input_dim: ${data_module.input_dim}
  output_dim: ${data_module.output_dim}
  encode_dim: 128
  rnn_num_layers: 2
  rnn_hidden_size: 256
  rnn_type: gru
  encoder_cfg:
    _target_: dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP
    hidden_sizes: [256, 256]
  pnn_decoder_cfg:
    # ... decoder config ...
```

### Converted TPNN Config
```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.tpnn.TPNN
  input_dim: ${data_module.input_dim}
  output_dim: ${data_module.output_dim}
  encode_dim: 128  # Keep same
  num_transformer_blocks: 4  # Roughly 2x rnn_num_layers
  num_heads: 8  # New parameter
  block_size: 100  # New parameter, set based on sequence length
  dropout: 0.1  # New parameter for regularization
  encoder_cfg:
    _target_: dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP
    hidden_sizes: [256, 256]  # Keep same
  pnn_decoder_cfg:
    # ... same decoder config ...
```

## 5. Loading Pretrained Weights

```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.tpnn.TPNN
  # ... other config ...
  load_dir: outputs/2024-03-04/14-30-46
  seed: 0
```

Or in code:
```python
model = TPNN(
    # ... config ...
    load_dir="path/to/checkpoint",
    seed=0,
)
```

## 6. Fine-tuning Only Specific Layers

```yaml
model:
  _target_: dynamics_toolbox.models.pl_models.sequential_models.tpnn.TPNN
  # ... other config ...
  freeze_all_but: "output_layer"  # Options: null, "logvar_net", "output_layer", "output_layer_block_4"
```

## 7. Different Loss Functions

```yaml
# NLL loss (default)
model:
  loss_fn_str: "NLL"

# MSE loss
model:
  loss_fn_str: "MSE"

# Combined NLL+MSE
model:
  loss_fn_str: "NLL+MSE"
  nll_wt: 1.0
  mse_wt: 0.5
```

## 8. Hyperparameter Guidelines

| Use Case | encode_dim | num_blocks | num_heads | block_size | dropout | lr |
|----------|-----------|------------|-----------|------------|---------|-----|
| Small dataset | 64 | 2 | 4 | 50 | 0.2 | 1e-3 |
| Medium dataset | 128 | 4 | 8 | 100 | 0.1 | 1e-4 |
| Large dataset | 256 | 6 | 8 | 200 | 0.1 | 1e-4 |

**Rules of thumb:**
- `encode_dim` must be divisible by `num_heads`
- `block_size` should be ≥ your longest sequence
- More heads = more expressive but more parameters
- Transformers typically need lower learning rates than RNNs

## 9. Common Issues and Solutions

### Issue: Out of memory
**Solution:** Reduce `block_size`, `encode_dim`, or `num_transformer_blocks`

### Issue: encode_dim not divisible by num_heads
**Solution:** Choose compatible values (e.g., encode_dim=128, num_heads=8)

### Issue: Model not learning
**Solution:** 
- Try lower learning rate (1e-4 to 1e-5)
- Increase `dropout` (0.1 to 0.3)
- Check that `block_size` ≥ sequence length

### Issue: Slower than RPNN
**Solution:** 
- TPNN is faster during training (parallel) but may be slower for single-step inference
- Reduce `block_size` if you don't need long history

## 10. Monitoring Training

Key metrics to watch:
- `loss`: Should decrease
- `mse/mean`: Mean squared error
- `nll/mean`: Negative log likelihood
- `logvar/mean`: Average log variance (should be reasonable, not too extreme)
- `EV`: Explained variance (should increase toward 1.0)

## 11. Example Training Script

```python
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Model
    model = hydra.utils.instantiate(cfg.model)
    
    # Data
    data_module = hydra.utils.instantiate(cfg.data_module)
    
    # Trainer
    trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Train
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    main()
```

## 12. Comparison Test

Run the comparison script to verify everything works:

```bash
cd /zfsauton2/home/rsonker/dynamics-toolbox
python scripts/compare_rpnn_tpnn.py
```

This will test:
- Interface equivalence with RPNN
- Sequential inference
- Batch processing
- All methods and properties

---

## Next Steps

1. ✅ Install/verify dependencies
2. ✅ Create your config file (or modify existing RPNN config)
3. ✅ Run comparison script to verify
4. ✅ Train your first TPNN model
5. ✅ Compare results with RPNN

For more details, see:
- `TPNN_README.md` - Full documentation
- `TPNN_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `example_configs/model/tpnn.yaml` - Example configuration
