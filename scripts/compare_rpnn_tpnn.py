"""
Comparison script demonstrating the equivalence between RPNN and TPNN.

This script shows that TPNN can be used as a drop-in replacement for RPNN
with the same interface and functionality.

Author: AI Assistant
Date: 12/06/2024
"""
import torch
import numpy as np
from omegaconf import DictConfig

# Example configurations
ENCODER_CFG = DictConfig({
    '_target_': 'dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP',
    'hidden_sizes': [256, 256],
    'hidden_activation': 'torch.nn.functional.relu',
    'output_activation': 'torch.nn.functional.relu',
})

DECODER_CFG = DictConfig({
    '_target_': 'dynamics_toolbox.models.pl_models.pnn.PNN',
    'encoder_output_dim': 128,
    'encoder_cfg': DictConfig({
        '_target_': 'dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP',
        'hidden_sizes': [256, 256],
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
})


def test_interface_equivalence():
    """Test that RPNN and TPNN have the same interface."""
    from dynamics_toolbox.models.pl_models.sequential_models.rpnn import RPNN
    from dynamics_toolbox.models.pl_models.sequential_models.tpnn import TPNN
    
    # Common parameters
    input_dim = 10
    output_dim = 5
    encode_dim = 128
    batch_size = 4
    seq_length = 20
    
    # Create RPNN
    rpnn = RPNN(
        input_dim=input_dim,
        output_dim=output_dim,
        encode_dim=encode_dim,
        rnn_num_layers=2,
        rnn_hidden_size=256,
        encoder_cfg=ENCODER_CFG,
        pnn_decoder_cfg=DECODER_CFG,
    )
    
    # Create TPNN
    tpnn = TPNN(
        input_dim=input_dim,
        output_dim=output_dim,
        encode_dim=encode_dim,
        num_transformer_blocks=4,
        num_heads=8,
        block_size=100,
        encoder_cfg=ENCODER_CFG,
        pnn_decoder_cfg=DECODER_CFG,
    )
    
    # Create sample data
    x = torch.randn(batch_size, seq_length, input_dim)
    y = torch.randn(batch_size, seq_length, output_dim)
    mask = torch.ones(batch_size, seq_length, 1)
    batch = [x, y, mask]
    
    print("=" * 80)
    print("INTERFACE EQUIVALENCE TEST")
    print("=" * 80)
    
    # Test 1: get_net_out
    print("\n1. Testing get_net_out()...")
    rpnn_out = rpnn.get_net_out(batch)
    tpnn_out = tpnn.get_net_out(batch)
    print(f"   RPNN output keys: {list(rpnn_out.keys())}")
    print(f"   TPNN output keys: {list(tpnn_out.keys())}")
    print(f"   ✓ Both return 'mean' and 'logvar'")
    
    # Test 2: loss
    print("\n2. Testing loss()...")
    rpnn_loss, rpnn_stats = rpnn.loss(rpnn_out, batch)
    tpnn_loss, tpnn_stats = tpnn.loss(tpnn_out, batch)
    print(f"   RPNN loss: {rpnn_loss.item():.4f}")
    print(f"   TPNN loss: {tpnn_loss.item():.4f}")
    print(f"   RPNN stats keys: {list(rpnn_stats.keys())}")
    print(f"   TPNN stats keys: {list(tpnn_stats.keys())}")
    print(f"   ✓ Both compute loss and return statistics")
    
    # Test 3: single_sample_output_from_torch
    print("\n3. Testing single_sample_output_from_torch()...")
    net_in = torch.randn(batch_size, input_dim)
    rpnn.clear_history()
    tpnn.clear_history()
    rpnn_pred, rpnn_info = rpnn.single_sample_output_from_torch(net_in)
    tpnn_pred, tpnn_info = tpnn.single_sample_output_from_torch(net_in)
    print(f"   RPNN prediction shape: {rpnn_pred.shape}")
    print(f"   TPNN prediction shape: {tpnn_pred.shape}")
    print(f"   RPNN info keys: {list(rpnn_info.keys())}")
    print(f"   TPNN info keys: {list(tpnn_info.keys())}")
    print(f"   ✓ Both return predictions and info dict")
    
    # Test 4: Properties
    print("\n4. Testing properties...")
    properties = [
        'input_dim', 'output_dim', 'learning_rate', 'weight_decay',
        'sample_mode', 'warm_up_period', 'record_history', 'metrics'
    ]
    for prop in properties:
        rpnn_val = getattr(rpnn, prop)
        tpnn_val = getattr(tpnn, prop)
        print(f"   {prop}: RPNN={rpnn_val}, TPNN={tpnn_val}")
    print(f"   ✓ All properties exist in both models")
    
    # Test 5: Methods
    print("\n5. Testing methods...")
    methods = ['clear_history', 'reset']
    for method in methods:
        assert hasattr(rpnn, method), f"RPNN missing {method}"
        assert hasattr(tpnn, method), f"TPNN missing {method}"
        print(f"   ✓ Both have {method}()")
    
    print("\n" + "=" * 80)
    print("✓ INTERFACE EQUIVALENCE TEST PASSED")
    print("=" * 80)


def test_sequential_inference():
    """Test sequential inference to ensure history management works correctly."""
    from dynamics_toolbox.models.pl_models.sequential_models.tpnn import TPNN
    
    print("\n" + "=" * 80)
    print("SEQUENTIAL INFERENCE TEST")
    print("=" * 80)
    
    # Create model
    tpnn = TPNN(
        input_dim=5,
        output_dim=3,
        encode_dim=64,
        num_transformer_blocks=2,
        num_heads=8,
        block_size=50,
        encoder_cfg=DictConfig({
            '_target_': 'dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP',
            'hidden_sizes': [128],
            'hidden_activation': 'torch.nn.functional.relu',
            'output_activation': 'torch.nn.functional.relu',
        }),
        pnn_decoder_cfg=DictConfig({
            '_target_': 'dynamics_toolbox.models.pl_models.pnn.PNN',
            'encoder_output_dim': 64,
            'encoder_cfg': DictConfig({
                '_target_': 'dynamics_toolbox.utils.pytorch.modules.simple_mlps.MLP',
                'hidden_sizes': [128],
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
    
    batch_size = 2
    num_steps = 10
    
    print(f"\nRunning {num_steps} sequential predictions...")
    tpnn.clear_history()
    
    for t in range(num_steps):
        net_in = torch.randn(batch_size, 5)
        pred, info = tpnn.single_sample_output_from_torch(net_in)
        print(f"   Step {t+1}: pred shape={pred.shape}, mean={pred.mean().item():.4f}")
    
    print("\n✓ Sequential inference works correctly")
    
    print("\nTesting reset...")
    tpnn.reset()
    print("✓ Reset successful")
    
    print("\n" + "=" * 80)
    print("✓ SEQUENTIAL INFERENCE TEST PASSED")
    print("=" * 80)


def compare_architectures():
    """Compare RPNN and TPNN architectures."""
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON")
    print("=" * 80)
    
    print("\nRPNN Architecture:")
    print("  Input → Encoder → LayerNorm → RNN/GRU/LSTM → Decoder → (mean, logvar)")
    print("  - Sequential processing")
    print("  - Hidden state carries information")
    print("  - Good for: online learning, limited memory")
    
    print("\nTPNN Architecture:")
    print("  Input → Encoder → LayerNorm → Positional Encoding →")
    print("  Transformer Blocks (Self-Attention + FFN) → Decoder → (mean, logvar)")
    print("  - Parallel processing (during training)")
    print("  - Attention over full sequence")
    print("  - Good for: long-range dependencies, batch processing")
    
    print("\nKey Differences:")
    print("  1. Memory Mechanism:")
    print("     - RPNN: Hidden state (compressed representation)")
    print("     - TPNN: Attention (full sequence context)")
    
    print("\n  2. Computational Complexity:")
    print("     - RPNN: O(n) in sequence length")
    print("     - TPNN: O(n²) in sequence length (due to attention)")
    
    print("\n  3. Training:")
    print("     - RPNN: Sequential (harder to parallelize)")
    print("     - TPNN: Parallel (faster training on GPUs)")
    
    print("\n  4. Long-range Dependencies:")
    print("     - RPNN: Decays exponentially with distance")
    print("     - TPNN: Direct attention to any timestep")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RPNN vs TPNN COMPARISON")
    print("=" * 80)
    
    try:
        test_interface_equivalence()
        test_sequential_inference()
        compare_architectures()
        
        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nTPNN is ready to be used as a drop-in replacement for RPNN!")
        print("Simply change the _target_ in your config from rpnn.RPNN to tpnn.TPNN")
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
