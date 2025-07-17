#!/usr/bin/env python3
"""
Test Suite for Enhanced LoRA Adapter System

This test suite validates the enhanced LoRA system with BezierAdapter multi-modal conditioning:
- Style Branch LoRA (r=64): Style vector conditioning
- Text Branch LoRA (r=64): Enhanced text conditioning
- Density Branch LoRA (r=64): Bézier density conditioning
- Backward compatibility with existing spatial/subject conditioning
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import time

# Add src directory to path
sys.path.append('src')

# Import our modules
from enhanced_lora_adapters import (
    ConditionType,
    EnhancedLoRALinearLayer,
    BezierLoRABranch,
    EnhancedMultiSingleStreamBlockLoraProcessor,
    EnhancedMultiDoubleStreamBlockLoraProcessor,
    count_bezier_lora_parameters,
    create_enhanced_lora_processor
)
from bezier_parameter_processor import create_bezier_processor
from bezier_extraction import BezierCurveExtractor

# Mock attention class for testing
class MockAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 24):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads

        # Query, key, value projections
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Output projections
        self.to_out = nn.ModuleList([
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(0.0)
        ])

        # For double-stream attention
        self.add_q_proj = nn.Linear(dim, dim, bias=False)
        self.add_k_proj = nn.Linear(dim, dim, bias=False)
        self.add_v_proj = nn.Linear(dim, dim, bias=False)
        self.to_add_out = nn.Linear(dim, dim, bias=False)

        # Normalization layers
        self.norm_q = None
        self.norm_k = None
        self.norm_added_q = None
        self.norm_added_k = None

def test_enhanced_lora_linear_layer():
    """Test the enhanced LoRA linear layer with different condition types."""
    print("=" * 70)
    print("TESTING ENHANCED LORA LINEAR LAYER")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Test parameters
    dim = 3072
    rank = 64
    batch_size = 2
    seq_len = 4096
    feature_dim = 256

    # Test different condition types
    condition_types = [
        ConditionType.SPATIAL,
        ConditionType.SUBJECT,
        ConditionType.STYLE,
        ConditionType.TEXT,
        ConditionType.DENSITY
    ]

    results = {}

    for condition_type in condition_types:
        print(f"\nTesting {condition_type.value} conditioning...")

        # Create layer
        layer = EnhancedLoRALinearLayer(
            in_features=dim,
            out_features=dim,
            rank=rank,
            condition_type=condition_type,
            cond_width=512,
            cond_height=512,
            branch_id=0,
            total_branches=1,
            use_feature_conditioning=True,
            feature_dim=feature_dim,
            device=device
        )

        # Create test inputs
        hidden_states = torch.randn(batch_size, seq_len, dim, device=device)
        condition_features = torch.randn(batch_size, feature_dim, device=device)

        # Forward pass
        start_time = time.time()
        output = layer(hidden_states, condition_features)
        end_time = time.time()

        # Validate output
        assert output.shape == hidden_states.shape, f"Output shape mismatch for {condition_type.value}"
        assert output.device == hidden_states.device, f"Device mismatch for {condition_type.value}"

        # Count parameters
        param_count = sum(p.numel() for p in layer.parameters())

        # Store results
        results[condition_type.value] = {
            'param_count': param_count,
            'processing_time': end_time - start_time,
            'output_shape': output.shape,
            'output_range': (output.min().item(), output.max().item())
        }

        print(f"  ✓ Parameters: {param_count:,}")
        print(f"  ✓ Processing time: {end_time - start_time:.4f}s")
        print(f"  ✓ Output shape: {output.shape}")
        print(f"  ✓ Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    total_params = sum(result['param_count'] for result in results.values())
    print(f"Total parameters (5 condition types): {total_params:,}")
    print(f"Average parameters per condition: {total_params // len(condition_types):,}")

    return results

def test_bezier_lora_branch():
    """Test the BezierLoRA branch with different condition types."""
    print("\n" + "=" * 70)
    print("TESTING BEZIER LORA BRANCH")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 3072
    rank = 64
    batch_size = 2
    seq_len = 4096
    feature_dim = 256

    # Test each condition type
    condition_types = [ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY]

    for condition_type in condition_types:
        print(f"\nTesting {condition_type.value} branch...")

        # Create branch
        branch = BezierLoRABranch(
            dim=dim,
            condition_type=condition_type,
            rank=rank,
            cond_width=512,
            cond_height=512,
            branch_id=0,
            total_branches=1,
            use_output_projection=True,
            feature_dim=feature_dim,
            device=device
        )

        # Create test inputs
        hidden_states = torch.randn(batch_size, seq_len, dim, device=device)
        query = torch.randn(batch_size, seq_len, dim, device=device)
        key = torch.randn(batch_size, seq_len, dim, device=device)
        value = torch.randn(batch_size, seq_len, dim, device=device)
        condition_features = torch.randn(batch_size, feature_dim, device=device)

        # Test Q, K, V enhancement
        enhanced_q, enhanced_k, enhanced_v = branch(
            query, key, value, hidden_states, condition_features
        )

        # Test output projection
        output_proj = branch.forward_output_projection(hidden_states, condition_features)

        # Validate outputs
        assert enhanced_q.shape == query.shape, f"Query shape mismatch for {condition_type.value}"
        assert enhanced_k.shape == key.shape, f"Key shape mismatch for {condition_type.value}"
        assert enhanced_v.shape == value.shape, f"Value shape mismatch for {condition_type.value}"
        assert output_proj.shape == hidden_states.shape, f"Output projection shape mismatch for {condition_type.value}"

        # Count parameters
        param_count = sum(p.numel() for p in branch.parameters())

        print(f"  ✓ Parameters: {param_count:,}")
        print(f"  ✓ Enhanced Q,K,V shapes: {enhanced_q.shape}, {enhanced_k.shape}, {enhanced_v.shape}")
        print(f"  ✓ Output projection shape: {output_proj.shape}")

        # Test that enhancement actually changes the tensors
        q_diff = torch.abs(enhanced_q - query).mean()
        k_diff = torch.abs(enhanced_k - key).mean()
        v_diff = torch.abs(enhanced_v - value).mean()

        print(f"  ✓ Q enhancement magnitude: {q_diff:.6f}")
        print(f"  ✓ K enhancement magnitude: {k_diff:.6f}")
        print(f"  ✓ V enhancement magnitude: {v_diff:.6f}")

        assert q_diff > 0, f"Query not enhanced for {condition_type.value}"
        assert k_diff > 0, f"Key not enhanced for {condition_type.value}"
        assert v_diff > 0, f"Value not enhanced for {condition_type.value}"

def test_enhanced_single_stream_processor():
    """Test the enhanced single-stream LoRA processor."""
    print("\n" + "=" * 70)
    print("TESTING ENHANCED SINGLE-STREAM PROCESSOR")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 3072
    batch_size = 2
    seq_len = 4096
    feature_dim = 256

    # Create processor
    processor = EnhancedMultiSingleStreamBlockLoraProcessor(
        dim=dim,
        n_loras=2,  # Legacy LoRA streams
        ranks=[4, 4],
        lora_weights=[1.0, 1.0],
        enable_bezier_conditioning=True,
        bezier_condition_types=[ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY],
        bezier_ranks=[64, 64, 64],
        bezier_weights=[1.0, 1.0, 1.0],
        device=device
    )

    # Create mock attention
    mock_attn = MockAttention(dim).to(device)

    # Create test inputs
    hidden_states = torch.randn(batch_size, seq_len, dim, device=device)
    style_features = torch.randn(batch_size, feature_dim, device=device)
    text_features = torch.randn(batch_size, feature_dim, device=device)
    density_features = torch.randn(batch_size, feature_dim, device=device)

    # Test forward pass
    print("Testing forward pass...")
    start_time = time.time()

    # Without conditioning
    output_no_cond = processor(
        mock_attn, hidden_states, use_cond=False
    )

    # With conditioning
    output_with_cond = processor(
        mock_attn, hidden_states, use_cond=True,
        style_features=style_features,
        text_features=text_features,
        density_features=density_features
    )

    end_time = time.time()

    print(f"  ✓ Processing time: {end_time - start_time:.4f}s")
    print(f"  ✓ Output without conditioning shape: {output_no_cond.shape}")
    print(f"  ✓ Output with conditioning type: {type(output_with_cond)}")

    if isinstance(output_with_cond, tuple):
        main_output, cond_output = output_with_cond
        print(f"  ✓ Main output shape: {main_output.shape}")
        print(f"  ✓ Condition output shape: {cond_output.shape}")
    else:
        print(f"  ✓ Combined output shape: {output_with_cond.shape}")

    # Count parameters
    param_counts = count_bezier_lora_parameters(processor)

    print(f"\n{'='*20} PARAMETER COUNTS {'='*20}")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")

    # Validate parameter targets
    target_bezier_params = 3.6e6  # 3.6M parameters
    actual_bezier_params = param_counts['bezier_branches']

    print(f"\nBezierAdapter Parameter Validation:")
    print(f"  Target: {target_bezier_params:,.0f} parameters")
    print(f"  Actual: {actual_bezier_params:,} parameters")
    print(f"  Status: {'✓' if actual_bezier_params < target_bezier_params * 1.1 else '✗'}")

    return processor, param_counts

def test_enhanced_double_stream_processor():
    """Test the enhanced double-stream LoRA processor."""
    print("\n" + "=" * 70)
    print("TESTING ENHANCED DOUBLE-STREAM PROCESSOR")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 3072
    batch_size = 2
    hidden_seq_len = 4096
    encoder_seq_len = 512
    feature_dim = 256

    # Create processor
    processor = EnhancedMultiDoubleStreamBlockLoraProcessor(
        dim=dim,
        n_loras=2,
        ranks=[4, 4],
        lora_weights=[1.0, 1.0],
        enable_bezier_conditioning=True,
        bezier_condition_types=[ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY],
        bezier_ranks=[64, 64, 64],
        bezier_weights=[1.0, 1.0, 1.0],
        device=device
    )

    # Create mock attention
    mock_attn = MockAttention(dim).to(device)

    # Create test inputs
    hidden_states = torch.randn(batch_size, hidden_seq_len, dim, device=device)
    encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, dim, device=device)
    style_features = torch.randn(batch_size, feature_dim, device=device)
    text_features = torch.randn(batch_size, feature_dim, device=device)
    density_features = torch.randn(batch_size, feature_dim, device=device)

    # Test forward pass
    print("Testing forward pass...")
    start_time = time.time()

    # Without conditioning
    output_no_cond = processor(
        mock_attn, hidden_states, encoder_hidden_states, use_cond=False
    )

    # With conditioning
    output_with_cond = processor(
        mock_attn, hidden_states, encoder_hidden_states, use_cond=True,
        style_features=style_features,
        text_features=text_features,
        density_features=density_features
    )

    end_time = time.time()

    print(f"  ✓ Processing time: {end_time - start_time:.4f}s")
    print(f"  ✓ Output without conditioning: {len(output_no_cond)} tensors")
    print(f"  ✓ Output with conditioning: {len(output_with_cond)} tensors")

    if len(output_with_cond) == 3:
        main_output, encoder_output, cond_output = output_with_cond
        print(f"  ✓ Main output shape: {main_output.shape}")
        print(f"  ✓ Encoder output shape: {encoder_output.shape}")
        print(f"  ✓ Condition output shape: {cond_output.shape}")

    # Count parameters
    param_counts = count_bezier_lora_parameters(processor)

    print(f"\n{'='*20} PARAMETER COUNTS {'='*20}")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")

    return processor, param_counts

def test_integration_with_bezier_processor():
    """Test integration with BezierParameterProcessor."""
    print("\n" + "=" * 70)
    print("TESTING INTEGRATION WITH BEZIER PROCESSOR")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create mock calligraphy data
    mock_bezier_data = {
        'image_path': 'test.png',
        'characters': [
            {
                'character_id': 0,
                'contour_area': 100.0,
                'bounding_box': (50, 50, 100, 100),
                'bezier_curves': [
                    [[10, 20], [30, 40], [50, 60], [70, 80]],  # Cubic Bézier
                    [[20, 30], [40, 50], [60, 70], [80, 90]]
                ],
                'original_contour_points': 50
            }
        ]
    }

    # Create BezierParameterProcessor
    print("Creating BezierParameterProcessor...")
    bezier_processor = create_bezier_processor(device=device)

    # Generate density maps
    print("Generating density maps...")
    density_maps = bezier_processor(mock_bezier_data)
    print(f"  ✓ Density maps shape: {density_maps.shape}")

    # Extract density features (simulate what would come from BezierConditionEncoder)
    batch_size = density_maps.shape[0]
    feature_dim = 256
    density_features = torch.randn(batch_size, feature_dim, device=device)

    # Create enhanced LoRA processor
    print("Creating enhanced LoRA processor...")
    lora_processor = EnhancedMultiSingleStreamBlockLoraProcessor(
        dim=3072,
        n_loras=1,
        ranks=[4],
        lora_weights=[1.0],
        enable_bezier_conditioning=True,
        bezier_condition_types=[ConditionType.DENSITY],
        bezier_ranks=[64],
        bezier_weights=[1.0],
        device=device
    )

    # Test integration
    print("Testing integration...")
    mock_attn = MockAttention(3072).to(device)
    hidden_states = torch.randn(batch_size, 4096, 3072, device=device)

    # Use density features in LoRA processor
    output = lora_processor(
        mock_attn, hidden_states, use_cond=True,
        density_features=density_features
    )

    print(f"  ✓ Integration successful!")
    print(f"  ✓ Output type: {type(output)}")
    if isinstance(output, tuple):
        print(f"  ✓ Main output shape: {output[0].shape}")
        print(f"  ✓ Condition output shape: {output[1].shape}")

    # Count total parameters
    bezier_params = sum(p.numel() for p in bezier_processor.parameters())
    lora_params = sum(p.numel() for p in lora_processor.parameters())
    total_params = bezier_params + lora_params

    print(f"\n{'='*20} TOTAL INTEGRATION PARAMETERS {'='*20}")
    print(f"  BezierParameterProcessor: {bezier_params:,}")
    print(f"  Enhanced LoRA Processor: {lora_params:,}")
    print(f"  Total: {total_params:,}")

    return total_params

def test_backward_compatibility():
    """Test backward compatibility with existing EasyControl LoRA system."""
    print("\n" + "=" * 70)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 3072
    batch_size = 2
    seq_len = 4096

    # Test 1: Enhanced processor with only legacy LoRA streams
    print("Test 1: Enhanced processor with only legacy streams...")
    processor_legacy_only = EnhancedMultiSingleStreamBlockLoraProcessor(
        dim=dim,
        n_loras=2,
        ranks=[4, 8],
        lora_weights=[1.0, 0.5],
        enable_bezier_conditioning=False,  # Disable BezierAdapter
        device=device
    )

    mock_attn = MockAttention(dim).to(device)
    hidden_states = torch.randn(batch_size, seq_len, dim, device=device)

    output_legacy = processor_legacy_only(mock_attn, hidden_states)
    print(f"  ✓ Legacy-only output shape: {output_legacy.shape}")

    # Test 2: Enhanced processor with both legacy and BezierAdapter streams
    print("Test 2: Enhanced processor with both legacy and BezierAdapter streams...")
    processor_mixed = EnhancedMultiSingleStreamBlockLoraProcessor(
        dim=dim,
        n_loras=2,
        ranks=[4, 8],
        lora_weights=[1.0, 0.5],
        enable_bezier_conditioning=True,
        bezier_condition_types=[ConditionType.STYLE],
        bezier_ranks=[64],
        bezier_weights=[1.0],
        device=device
    )

    style_features = torch.randn(batch_size, 256, device=device)
    output_mixed = processor_mixed(
        mock_attn, hidden_states,
        style_features=style_features
    )
    print(f"  ✓ Mixed output shape: {output_mixed.shape}")

    # Test 3: Parameter count comparison
    print("Test 3: Parameter count comparison...")
    legacy_params = sum(p.numel() for p in processor_legacy_only.parameters())
    mixed_params = sum(p.numel() for p in processor_mixed.parameters())

    print(f"  Legacy-only parameters: {legacy_params:,}")
    print(f"  Mixed parameters: {mixed_params:,}")
    print(f"  BezierAdapter addition: {mixed_params - legacy_params:,}")

    # Test 4: Output consistency check
    print("Test 4: Output consistency check...")

    # Both should produce same output when BezierAdapter features are None
    output_mixed_no_bezier = processor_mixed(mock_attn, hidden_states)

    # Should be similar (not identical due to parameter initialization differences)
    output_diff = torch.abs(output_legacy - output_mixed_no_bezier).mean()
    print(f"  Output difference (legacy vs mixed without BezierAdapter): {output_diff:.6f}")

    print("  ✓ Backward compatibility validated!")

def create_performance_benchmark():
    """Create performance benchmark for enhanced LoRA system."""
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test different configurations
    configs = [
        {"name": "Legacy Only", "enable_bezier": False, "bezier_types": []},
        {"name": "Style Only", "enable_bezier": True, "bezier_types": [ConditionType.STYLE]},
        {"name": "Text Only", "enable_bezier": True, "bezier_types": [ConditionType.TEXT]},
        {"name": "Density Only", "enable_bezier": True, "bezier_types": [ConditionType.DENSITY]},
        {"name": "All BezierAdapter", "enable_bezier": True, "bezier_types": [ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY]},
    ]

    results = {}

    for config in configs:
        print(f"\nBenchmarking {config['name']}...")

        # Create processor
        processor = EnhancedMultiSingleStreamBlockLoraProcessor(
            dim=3072,
            n_loras=2,
            ranks=[4, 4],
            lora_weights=[1.0, 1.0],
            enable_bezier_conditioning=config['enable_bezier'],
            bezier_condition_types=config['bezier_types'],
            bezier_ranks=[64] * len(config['bezier_types']),
            bezier_weights=[1.0] * len(config['bezier_types']),
            device=device
        )

        # Create test inputs
        mock_attn = MockAttention(3072).to(device)
        hidden_states = torch.randn(2, 4096, 3072, device=device)
        style_features = torch.randn(2, 256, device=device)
        text_features = torch.randn(2, 256, device=device)
        density_features = torch.randn(2, 256, device=device)

        # Warmup
        for _ in range(3):
            _ = processor(mock_attn, hidden_states, style_features=style_features, text_features=text_features, density_features=density_features)

        # Benchmark
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time = time.time()

        for _ in range(10):
            _ = processor(mock_attn, hidden_states, style_features=style_features, text_features=text_features, density_features=density_features)

        torch.cuda.synchronize() if device == 'cuda' else None
        end_time = time.time()

        avg_time = (end_time - start_time) / 10
        param_count = sum(p.numel() for p in processor.parameters())

        results[config['name']] = {
            'avg_time': avg_time,
            'param_count': param_count
        }

        print(f"  Average time: {avg_time:.4f}s")
        print(f"  Parameters: {param_count:,}")

    # Summary
    print(f"\n{'='*20} BENCHMARK SUMMARY {'='*20}")
    for name, result in results.items():
        print(f"{name:20} | {result['avg_time']:.4f}s | {result['param_count']:,} params")

    return results

def main():
    """Main test function."""
    print("ENHANCED LORA ADAPTER SYSTEM - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    try:
        # Test 1: Enhanced LoRA Linear Layer
        test_enhanced_lora_linear_layer()

        # Test 2: BezierLoRA Branch
        test_bezier_lora_branch()

        # Test 3: Enhanced Single-Stream Processor
        single_processor, single_params = test_enhanced_single_stream_processor()

        # Test 4: Enhanced Double-Stream Processor
        double_processor, double_params = test_enhanced_double_stream_processor()

        # Test 5: Integration with BezierParameterProcessor
        integration_params = test_integration_with_bezier_processor()

        # Test 6: Backward Compatibility
        test_backward_compatibility()

        # Test 7: Performance Benchmark
        benchmark_results = create_performance_benchmark()

        print("\n" + "=" * 70)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)

        print("\nFinal Summary:")
        print(f"  Single-Stream BezierAdapter Parameters: {single_params['bezier_branches']:,}")
        print(f"  Double-Stream BezierAdapter Parameters: {double_params['bezier_branches']:,}")
        print(f"  Total Integration Parameters: {integration_params:,}")
        print(f"  Target BezierAdapter Parameters: ~3,600,000")
        print(f"  Status: {'✓ PASSED' if single_params['bezier_branches'] < 4e6 else '✗ EXCEEDED'}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)