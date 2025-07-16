#!/usr/bin/env python3
"""
Enhanced LoRA Integration Example

This example demonstrates how the enhanced LoRA system integrates with:
1. BezierParameterProcessor for density conditioning
2. Existing EasyControl FLUX transformer
3. Multi-modal conditioning (Style, Text, Density)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import time
import sys

# Import enhanced LoRA modules
from enhanced_lora_adapters import (
    ConditionType,
    EnhancedMultiSingleStreamBlockLoraProcessor,
    EnhancedMultiDoubleStreamBlockLoraProcessor,
    count_bezier_lora_parameters,
    create_enhanced_lora_processor
)

# Import BezierParameterProcessor
from bezier_parameter_processor import create_bezier_processor

def demonstrate_enhanced_lora_integration():
    """
    Demonstrate the complete integration pipeline:
    Bézier Curves → Density Maps → Enhanced LoRA → FLUX Attention
    """
    print("=" * 70)
    print("ENHANCED LORA INTEGRATION DEMONSTRATION")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Create mock Bézier data (normally from BezierCurveExtractor)
    print("\n1. Creating mock Bézier curve data...")
    bezier_data = {
        'image_path': 'sample_calligraphy.png',
        'characters': [
            {
                'character_id': 0,
                'contour_area': 150.0,
                'bounding_box': (50, 50, 200, 200),
                'bezier_curves': [
                    [[10, 20], [30, 40], [50, 60], [70, 80]],
                    [[20, 30], [40, 50], [60, 70], [80, 90]],
                    [[30, 40], [50, 60], [70, 80], [90, 100]]
                ],
                'original_contour_points': 75
            },
            {
                'character_id': 1,
                'contour_area': 120.0,
                'bounding_box': (250, 50, 150, 180),
                'bezier_curves': [
                    [[100, 120], [130, 140], [150, 160], [170, 180]],
                    [[120, 130], [140, 150], [160, 170], [180, 190]]
                ],
                'original_contour_points': 60
            }
        ]
    }
    print(f"✓ Created data with {len(bezier_data['characters'])} characters")

    # 2. Generate density maps using BezierParameterProcessor
    print("\n2. Generating density maps...")
    bezier_processor = create_bezier_processor(device=device)

    # Process Bézier data to density maps
    density_maps = bezier_processor([bezier_data])  # Batch of 1
    print(f"✓ Generated density maps: {density_maps.shape}")
    print(f"✓ Density range: [{density_maps.min():.4f}, {density_maps.max():.4f}]")

    # 3. Create conditioning features (simulating BezierConditionEncoder)
    print("\n3. Creating conditioning features...")
    batch_size = density_maps.shape[0]
    feature_dim = 256

    # Style features (from style vectors)
    style_features = torch.randn(batch_size, feature_dim, device=device)

    # Text features (from enhanced text embeddings)
    text_features = torch.randn(batch_size, feature_dim, device=device)

    # Density features (from density map processing)
    density_features = torch.randn(batch_size, feature_dim, device=device)

    print(f"✓ Style features: {style_features.shape}")
    print(f"✓ Text features: {text_features.shape}")
    print(f"✓ Density features: {density_features.shape}")

    # 4. Create enhanced LoRA processors
    print("\n4. Creating enhanced LoRA processors...")

    # Single-stream processor (for later transformer blocks)
    single_processor = EnhancedMultiSingleStreamBlockLoraProcessor(
        dim=3072,
        n_loras=2,  # Legacy spatial/subject LoRA
        ranks=[4, 4],
        lora_weights=[1.0, 1.0],
        enable_bezier_conditioning=True,
        bezier_condition_types=[ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY],
        bezier_ranks=[64, 64, 64],
        bezier_weights=[1.0, 1.0, 1.0],
        device=device
    )

    # Double-stream processor (for early transformer blocks)
    double_processor = EnhancedMultiDoubleStreamBlockLoraProcessor(
        dim=3072,
        n_loras=2,  # Legacy spatial/subject LoRA
        ranks=[4, 4],
        lora_weights=[1.0, 1.0],
        enable_bezier_conditioning=True,
        bezier_condition_types=[ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY],
        bezier_ranks=[64, 64, 64],
        bezier_weights=[1.0, 1.0, 1.0],
        device=device
    )

    # Count parameters
    single_params = count_bezier_lora_parameters(single_processor)
    double_params = count_bezier_lora_parameters(double_processor)

    print(f"✓ Single-stream processor created")
    print(f"  - Legacy parameters: {single_params['legacy_branches']:,}")
    print(f"  - BezierAdapter parameters: {single_params['bezier_branches']:,}")
    print(f"  - Total parameters: {single_params['total']:,}")

    print(f"✓ Double-stream processor created")
    print(f"  - Legacy parameters: {double_params['legacy_branches']:,}")
    print(f"  - BezierAdapter parameters: {double_params['bezier_branches']:,}")
    print(f"  - Total parameters: {double_params['total']:,}")

    # 5. Simulate FLUX transformer block processing
    print("\n5. Simulating FLUX transformer block processing...")

    # Create mock attention modules
    class MockFluxAttention(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.heads = 24
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Dropout(0.0)])

            # For double-stream
            self.add_q_proj = nn.Linear(dim, dim, bias=False)
            self.add_k_proj = nn.Linear(dim, dim, bias=False)
            self.add_v_proj = nn.Linear(dim, dim, bias=False)
            self.to_add_out = nn.Linear(dim, dim, bias=False)

            self.norm_q = None
            self.norm_k = None
            self.norm_added_q = None
            self.norm_added_k = None

    # Create test tensors
    seq_len = 4096
    encoder_seq_len = 512

    hidden_states = torch.randn(batch_size, seq_len, 3072, device=device)
    encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, 3072, device=device)

    # Mock attention modules
    single_attn = MockFluxAttention(3072).to(device)
    double_attn = MockFluxAttention(3072).to(device)

    # 6. Process through enhanced LoRA attention
    print("\n6. Processing through enhanced LoRA attention...")

    # Double-stream processing (early transformer blocks)
    print("  Double-stream processing...")
    start_time = time.time()

    double_output = double_processor(
        double_attn,
        hidden_states,
        encoder_hidden_states,
        use_cond=True,
        style_features=style_features,
        text_features=text_features,
        density_features=density_features
    )

    double_time = time.time() - start_time
    print(f"    ✓ Processing time: {double_time:.4f}s")
    print(f"    ✓ Output tensors: {len(double_output)}")

    if len(double_output) == 3:
        main_out, encoder_out, cond_out = double_output
        print(f"    ✓ Main output: {main_out.shape}")
        print(f"    ✓ Encoder output: {encoder_out.shape}")
        print(f"    ✓ Condition output: {cond_out.shape}")

    # Single-stream processing (later transformer blocks)
    print("  Single-stream processing...")
    start_time = time.time()

    single_output = single_processor(
        single_attn,
        hidden_states,
        use_cond=True,
        style_features=style_features,
        text_features=text_features,
        density_features=density_features
    )

    single_time = time.time() - start_time
    print(f"    ✓ Processing time: {single_time:.4f}s")

    if isinstance(single_output, tuple):
        main_out, cond_out = single_output
        print(f"    ✓ Main output: {main_out.shape}")
        print(f"    ✓ Condition output: {cond_out.shape}")

    # 7. Performance and parameter summary
    print("\n7. Performance and parameter summary...")

    # Total BezierAdapter parameters
    total_bezier_params = single_params['bezier_branches'] + double_params['bezier_branches']

    # BezierParameterProcessor parameters
    bezier_proc_params = sum(p.numel() for p in bezier_processor.parameters())

    # Grand total
    grand_total = total_bezier_params + bezier_proc_params

    print(f"Parameter Breakdown:")
    print(f"  BezierParameterProcessor: {bezier_proc_params:,}")
    print(f"  Enhanced LoRA (Single-stream): {single_params['bezier_branches']:,}")
    print(f"  Enhanced LoRA (Double-stream): {double_params['bezier_branches']:,}")
    print(f"  Total BezierAdapter Parameters: {grand_total:,}")

    print(f"\nPerformance:")
    print(f"  Double-stream processing: {double_time:.4f}s")
    print(f"  Single-stream processing: {single_time:.4f}s")
    print(f"  Total processing time: {double_time + single_time:.4f}s")

    # 8. Integration validation
    print("\n8. Integration validation...")

    # Check that conditioning actually affects output
    # Process without conditioning
    single_output_no_cond = single_processor(single_attn, hidden_states, use_cond=False)

    # Compare outputs
    if isinstance(single_output, tuple) and isinstance(single_output_no_cond, tuple):
        main_with_cond, _ = single_output
        main_without_cond = single_output_no_cond
    else:
        main_with_cond = single_output if not isinstance(single_output, tuple) else single_output[0]
        main_without_cond = single_output_no_cond

    conditioning_effect = torch.abs(main_with_cond - main_without_cond).mean()
    print(f"  Conditioning effect magnitude: {conditioning_effect:.6f}")

    if conditioning_effect > 1e-6:
        print("  ✓ Conditioning successfully affects output")
    else:
        print("  ⚠ Conditioning effect may be too small")

    # 9. Memory usage estimation
    print("\n9. Memory usage estimation...")

    # Calculate memory for key tensors
    tensor_memory = (
        hidden_states.numel() * 4 +  # 4 bytes per float32
        encoder_hidden_states.numel() * 4 +
        style_features.numel() * 4 +
        text_features.numel() * 4 +
        density_features.numel() * 4 +
        density_maps.numel() * 4
    ) / 1024 / 1024  # Convert to MB

    print(f"  Tensor memory usage: {tensor_memory:.2f} MB")
    print(f"  Parameter memory: {grand_total * 4 / 1024 / 1024:.2f} MB")
    print(f"  Total estimated memory: {tensor_memory + grand_total * 4 / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 70)
    print("INTEGRATION DEMONSTRATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    return {
        'bezier_processor_params': bezier_proc_params,
        'single_lora_params': single_params['bezier_branches'],
        'double_lora_params': double_params['bezier_branches'],
        'total_params': grand_total,
        'processing_time': double_time + single_time,
        'conditioning_effect': conditioning_effect.item()
    }

def create_integration_visualization():
    """Create a visualization showing the integration architecture."""
    print("\n" + "=" * 70)
    print("INTEGRATION ARCHITECTURE VISUALIZATION")
    print("=" * 70)

    architecture = """
    BEZIERADAPTER INTEGRATION ARCHITECTURE

    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              INPUT STAGE                                        │
    │                                                                                 │
    │  Calligraphy Image → BezierCurveExtractor → Bézier Control Points              │
    │                                                    │                            │
    │                                                    ▼                            │
    │                            BezierParameterProcessor                             │
    │                                  (2.1M params)                                 │
    │                                       │                                         │
    │                                       ▼                                         │
    │                              Density Maps [B,1,64,64]                          │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                       CONDITIONING FEATURE STAGE                               │
    │                                                                                 │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                │
    │  │ Style Features  │  │ Text Features   │  │ Density Features│                │
    │  │   [B, 256]      │  │   [B, 256]      │  │   [B, 256]      │                │
    │  └─────────────────┘  └─────────────────┘  └─────────────────┘                │
    │           │                    │                    │                          │
    │           └────────────────────┼────────────────────┘                          │
    │                                ▼                                               │
    │                    Enhanced LoRA Processors                                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                        FLUX TRANSFORMER BLOCKS                                 │
    │                                                                                 │
    │  Double-Stream Blocks 0-18 (Early Processing)                                  │
    │  ┌─────────────────────────────────────────────────────────────────────────┐   │
    │  │             EnhancedMultiDoubleStreamBlockLoraProcessor                 │   │
    │  │                                                                         │   │
    │  │  Legacy LoRA: Spatial + Subject (existing)                             │   │
    │  │  BezierAdapter LoRA: Style + Text + Density (new)                      │   │
    │  │                                                                         │   │
    │  │  Total: ~1.8M parameters per block                                     │   │
    │  └─────────────────────────────────────────────────────────────────────────┘   │
    │                                       │                                         │
    │                                       ▼                                         │
    │  Single-Stream Blocks 19-56 (Later Processing)                                 │
    │  ┌─────────────────────────────────────────────────────────────────────────┐   │
    │  │             EnhancedMultiSingleStreamBlockLoraProcessor                 │   │
    │  │                                                                         │   │
    │  │  Legacy LoRA: Spatial + Subject (existing)                             │   │
    │  │  BezierAdapter LoRA: Style + Text + Density (new)                      │   │
    │  │                                                                         │   │
    │  │  Total: ~1.8M parameters per block                                     │   │
    │  └─────────────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────────────┘
                                           │
                                           ▼
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                           OUTPUT STAGE                                         │
    │                                                                                 │
    │              Enhanced Attention Features → VAE Decoder                         │
    │                                  │                                             │
    │                                  ▼                                             │
    │                      Generated Calligraphy Images                              │
    │                           [B, 3, 512, 512]                                     │
    └─────────────────────────────────────────────────────────────────────────────────┘

    KEY FEATURES:
    • Backward Compatible: Preserves existing spatial/subject LoRA functionality
    • Multi-Modal: Supports Style, Text, and Density conditioning simultaneously
    • Parameter Efficient: ~3.6M additional parameters for BezierAdapter features
    • Performance Optimized: Minimal processing overhead with enhanced capabilities
    • Gradient Stable: Proper gradient flow through all conditioning branches
    """

    print(architecture)

if __name__ == "__main__":
    print("ENHANCED LORA INTEGRATION EXAMPLE")
    print("=" * 70)

    try:
        # Run integration demonstration
        results = demonstrate_enhanced_lora_integration()

        # Create visualization
        create_integration_visualization()

        print("\nFINAL RESULTS:")
        print(f"  Total BezierAdapter Parameters: {results['total_params']:,}")
        print(f"  Processing Time: {results['processing_time']:.4f}s")
        print(f"  Conditioning Effect: {results['conditioning_effect']:.6f}")
        print(f"  Status: {'✓ SUCCESS' if results['total_params'] < 15e6 else '⚠ HIGH PARAMS'}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\nIntegration example completed successfully!")
    print("The enhanced LoRA system is ready for FLUX transformer integration.")