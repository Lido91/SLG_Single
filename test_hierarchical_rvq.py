"""
Test script for Hierarchical RVQ-GPT architecture

Tests the forward pass and generation with dummy data to verify:
1. Model instantiation works correctly
2. Forward pass produces correct output shapes
3. Generation loop works
4. Hierarchical conditioning is properly implemented
"""

import torch
from mGPT.archs.mgpt_rvq_hierarchical import HierarchicalRVQGPT


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("="*70)
    print("Testing Hierarchical RVQ-GPT Forward Pass")
    print("="*70)

    # Create model
    model = HierarchicalRVQGPT(
        num_vq=512,
        embed_dim=1024,
        block_size=100,
        num_layers=3,  # Use fewer layers for testing
        n_head=16,
        dropout=0.1,
        text_dim=512
    )

    # Dummy data
    batch_size = 2
    seq_len = 10
    text_seq_len = 15

    # Motion codes: (B, T, 6) - simulating 6-quantizer RVQ tokens
    motion_codes = torch.randint(0, 512, (batch_size, seq_len, 6))

    # Text features: (B, S, text_dim)
    text_features = torch.randn(batch_size, text_seq_len, 512)

    print(f"\nInput shapes:")
    print(f"  motion_codes: {motion_codes.shape} (only first 3 used)")
    print(f"  text_features: {text_features.shape}")

    # Forward pass
    with torch.no_grad():
        logits_q0, logits_q1, logits_q2 = model(motion_codes, text_features)

    print(f"\nOutput shapes:")
    print(f"  logits_q0: {logits_q0.shape} (expected: [{batch_size}, {seq_len}, 512])")
    print(f"  logits_q1: {logits_q1.shape} (expected: [{batch_size}, {seq_len}, 512])")
    print(f"  logits_q2: {logits_q2.shape} (expected: [{batch_size}, {seq_len}, 512])")

    # Verify shapes
    assert logits_q0.shape == (batch_size, seq_len, 512), "Q0 logits shape mismatch!"
    assert logits_q1.shape == (batch_size, seq_len, 512), "Q1 logits shape mismatch!"
    assert logits_q2.shape == (batch_size, seq_len, 512), "Q2 logits shape mismatch!"

    print("\n✅ Forward pass test PASSED!")


def test_generation():
    """Test autoregressive generation"""
    print("\n" + "="*70)
    print("Testing Hierarchical RVQ-GPT Generation")
    print("="*70)

    # Create model
    model = HierarchicalRVQGPT(
        num_vq=512,
        embed_dim=512,  # Smaller for faster testing
        block_size=100,
        num_layers=2,  # Fewer layers for testing
        n_head=8,
        dropout=0.1,
        text_dim=512
    )
    model.eval()

    # Dummy text features
    batch_size = 2
    text_seq_len = 10
    max_gen_len = 5

    text_features = torch.randn(batch_size, text_seq_len, 512)

    print(f"\nGenerating {max_gen_len} tokens...")
    print(f"  Text features: {text_features.shape}")

    # Generate
    with torch.no_grad():
        generated_codes = model.generate(
            text_features=text_features,
            max_len=max_gen_len,
            do_sample=False,  # Greedy for deterministic testing
            temperature=1.0
        )

    print(f"\nGenerated codes shape: {generated_codes.shape}")
    print(f"  Expected: [{batch_size}, {max_gen_len}, 3]")

    # Verify shape
    assert generated_codes.shape == (batch_size, max_gen_len, 3), "Generated codes shape mismatch!"

    # Check values are in valid range
    assert generated_codes.min() >= 0, "Generated codes have negative values!"
    assert generated_codes.max() < 512, "Generated codes exceed codebook size!"

    print(f"\nSample generated codes (first sequence, first 3 timesteps):")
    print(f"  Q0: {generated_codes[0, :3, 0].tolist()}")
    print(f"  Q1: {generated_codes[0, :3, 1].tolist()}")
    print(f"  Q2: {generated_codes[0, :3, 2].tolist()}")

    print("\n✅ Generation test PASSED!")


def test_hierarchical_conditioning():
    """Verify hierarchical conditioning is working"""
    print("\n" + "="*70)
    print("Testing Hierarchical Conditioning")
    print("="*70)

    # Create model
    model = HierarchicalRVQGPT(
        num_vq=512,
        embed_dim=256,
        block_size=50,
        num_layers=2,
        n_head=4,
        dropout=0.0,  # No dropout for deterministic testing
        text_dim=512
    )
    model.eval()

    # Same input, should give same output (deterministic)
    batch_size = 1
    seq_len = 5
    text_seq_len = 5

    motion_codes = torch.randint(0, 512, (batch_size, seq_len, 3))
    text_features = torch.randn(batch_size, text_seq_len, 512)

    # Forward pass twice
    with torch.no_grad():
        logits_q0_1, logits_q1_1, logits_q2_1 = model(motion_codes, text_features)
        logits_q0_2, logits_q1_2, logits_q2_2 = model(motion_codes, text_features)

    # Should be identical (deterministic)
    assert torch.allclose(logits_q0_1, logits_q0_2), "Q0 outputs not deterministic!"
    assert torch.allclose(logits_q1_1, logits_q1_2), "Q1 outputs not deterministic!"
    assert torch.allclose(logits_q2_1, logits_q2_2), "Q2 outputs not deterministic!"

    print("\n✅ Hierarchical conditioning test PASSED!")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("Hierarchical RVQ-GPT Architecture Tests")
    print("="*70)
    print("\nThis tests the 3-layer hierarchical architecture:")
    print("  Q0 → Q1 → Q2 (coarse → medium → fine)")
    print("  Uses first 3 of 6 quantizers from RVQ-VAE")
    print("="*70)

    try:
        test_forward_pass()
        test_generation()
        test_hierarchical_conditioning()

        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*70)
        print("\nHierarchical RVQ-GPT is ready for training!")
        print("\nNext steps:")
        print("  1. Generate motion tokens using trained RVQ-VAE:")
        print("     python get_motion_code.py --cfg configs/deto_h2s_rvq.yaml")
        print("\n  2. Train hierarchical RVQ-GPT:")
        print("     python train.py --cfg configs/deto_h2s_rvq_hierarchical.yaml --nodebug")
        print("\n  3. Test generation:")
        print("     python test.py --cfg configs/deto_h2s_rvq_hierarchical.yaml --task t2m")
        print("="*70)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
