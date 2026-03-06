"""
Validation script to verify that RPNNEncoderGRU output matches RPNN hidden state.

This script loads both the full RPNN model and the RPNNEncoderGRU model,
runs the same input through both, and verifies that the hidden states match.
"""

import torch
import numpy as np
from dynamics_toolbox.models.pl_models.sequential_models.rpnn_encoder_gru import RPNNEncoderGRU
from dynamics_toolbox.utils.storage.model_storage import load_model_from_log_dir


def validate_encoder_gru_matches_rpnn(
    rpnn_checkpoint_dir: str,
    input_dim: int,
    encode_dim: int,
    rnn_num_layers: int,
    rnn_hidden_size: int,
    seed: int = 0,
    num_test_samples: int = 10,
    tolerance: float = 1e-5,
):
    """
    Validate that RPNNEncoderGRU produces the same hidden states as full RPNN.
    
    Args:
        rpnn_checkpoint_dir: Path to RPNN checkpoint directory
        input_dim: Input dimension
        encode_dim: Encoder output dimension
        rnn_num_layers: Number of GRU layers
        rnn_hidden_size: GRU hidden size
        seed: Which seed checkpoint to load
        num_test_samples: Number of test samples to validate
        tolerance: Maximum allowed difference between outputs
    
    Returns:
        True if validation passes, False otherwise
    """
    print("="*70)
    print("VALIDATING: RPNNEncoderGRU vs Full RPNN")
    print("="*70)
    
    # Load full RPNN model
    print("\n1. Loading full RPNN model...")
    try:
        import os
        # Build path to specific seed directory
        rpnn_seed_dir = os.path.join(rpnn_checkpoint_dir, str(seed))
        rpnn_model = load_model_from_log_dir(rpnn_seed_dir)
        rpnn_model.eval()
        print(f"   ✓ Full RPNN model loaded from {rpnn_seed_dir}")
    except Exception as e:
        print(f"   ✗ Failed to load RPNN model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Load RPNNEncoderGRU model
    print("\n2. Loading RPNNEncoderGRU model...")
    try:
        encoder_gru = RPNNEncoderGRU(
            input_dim=input_dim,
            encode_dim=encode_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_hidden_size=rnn_hidden_size,
            rnn_type='gru',
            use_layer_norm=True,
            load_from_rpnn=rpnn_checkpoint_dir,
            seed=seed,
            device='cpu',
        )
        encoder_gru.eval()
        print("   ✓ RPNNEncoderGRU model loaded")
    except Exception as e:
        print(f"   ✗ Failed to load RPNNEncoderGRU: {e}")
        return False
    
    # Run validation tests
    print(f"\n3. Running validation with {num_test_samples} test samples...")
    
    max_diff = 0.0
    all_passed = True
    
    for i in range(num_test_samples):
        # Generate random input
        batch_size = np.random.randint(1, 5)
        test_input = torch.randn(batch_size, input_dim)
        
        # Get output from full RPNN (just the hidden state after encoder+GRU)
        with torch.no_grad():
            # Reset both models
            rpnn_model.clear_history()
            encoder_gru.clear_history()
            
            # RPNN forward pass - extract hidden state
            # The RPNN processes: input -> encoder -> layer_norm -> GRU -> decoder
            # We want the GRU output (before decoder)
            encoded = rpnn_model._encoder(test_input)
            if rpnn_model._use_layer_norm:
                encoded = rpnn_model._layer_norm(encoded)
            encoded_unsqueezed = encoded.unsqueeze(1)  # Add sequence dimension
            rpnn_hidden_state = rpnn_model._memory_unit(encoded_unsqueezed)[0]
            rpnn_hidden_state = rpnn_hidden_state.squeeze(1)  # Remove sequence dimension
            
            # Get output from RPNNEncoderGRU
            encoder_gru_output = encoder_gru.forward(test_input)
        
        # Compare outputs
        diff = torch.abs(rpnn_hidden_state - encoder_gru_output).max().item()
        max_diff = max(max_diff, diff)
        
        if diff > tolerance:
            print(f"   ✗ Test {i+1}/{num_test_samples}: FAILED (max diff = {diff:.2e})")
            all_passed = False
        else:
            print(f"   ✓ Test {i+1}/{num_test_samples}: PASSED (max diff = {diff:.2e})")
    
    # Print summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Maximum difference across all tests: {max_diff:.2e}")
    print(f"Tolerance threshold: {tolerance:.2e}")
    
    if all_passed:
        print("\n✓ SUCCESS! RPNNEncoderGRU matches RPNN hidden states perfectly!")
        print("  The extracted encoder+GRU produces identical outputs to the full RPNN.")
    else:
        print("\n✗ VALIDATION FAILED!")
        print("  RPNNEncoderGRU outputs do not match RPNN hidden states.")
        print("  There may be an issue with weight extraction or architecture.")
    
    print("="*70)
    
    return all_passed


if __name__ == "__main__":
    # Configuration for your RPNN model
    rpnn_checkpoint_dir = '/home/scratch/rsonker/dynamics_models/rpnn_noshape_gas_bms_step_one_mse_1f'
    
    # Run validation
    success = validate_encoder_gru_matches_rpnn(
        rpnn_checkpoint_dir=rpnn_checkpoint_dir,
        input_dim=55,
        encode_dim=512,
        rnn_num_layers=1,
        rnn_hidden_size=256,
        seed=0,
        num_test_samples=10,
        tolerance=1e-5,
    )
    
    exit(0 if success else 1)
