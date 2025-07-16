"""
Spatial Attention Integration Example

This example demonstrates the complete integration of SpatialAttentionFuser
with the FLUX transformer for BezierAdapter functionality.

Components demonstrated:
1. SpatialAttentionFuser (~3.8M parameters)
2. Integration with FLUX transformer blocks
3. BezierParameterProcessor → SpatialAttentionFuser pipeline
4. Complete end-to-end workflow

Architecture Integration:
- Phase 1: Double-Stream Blocks 12-18 (Enhanced Cross-Attention)
- Phase 3: Single-Stream Blocks 20-37 (Spatial Attention Enhancement)
"""

import torch
import torch.nn as nn
import numpy as np
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import sys
import os

# Add paths for imports
sys.path.append(os.path.dirname(__file__))

from spatial_attention_fuser import SpatialAttentionFuser, SpatialAttentionProcessor
from spatial_transformer_integration import SpatialTransformerIntegrator, BezierSpatialPipeline, validate_spatial_integration
from bezier_parameter_processor import BezierParameterProcessor
from enhanced_lora_adapters import EnhancedMultiSingleStreamBlockLoraProcessor

# Mock FLUX transformer for demonstration
class MockFluxTransformer(nn.Module):
    """Mock FLUX transformer for demonstration purposes."""

    def __init__(self, hidden_dim=3072, num_heads=24):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Create mock transformer blocks
        self.transformer_blocks = nn.ModuleList([
            self._create_mock_block() for _ in range(19)  # Double-stream blocks
        ])

        self.single_transformer_blocks = nn.ModuleList([
            self._create_mock_block() for _ in range(38)  # Single-stream blocks
        ])

        # Initialize attention processors
        self.attn_processors = {}
        for i in range(19):
            self.attn_processors[f'transformer_blocks.{i}.attn.processor'] = None
        for i in range(38):
            self.attn_processors[f'single_transformer_blocks.{i}.attn.processor'] = None

    def _create_mock_block(self):
        """Create a mock transformer block."""
        return nn.ModuleDict({
            'attn': nn.ModuleDict({
                'processor': None,
                'to_q': nn.Linear(self.hidden_dim, self.hidden_dim),
                'to_k': nn.Linear(self.hidden_dim, self.hidden_dim),
                'to_v': nn.Linear(self.hidden_dim, self.hidden_dim),
                'to_out': nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim)])
            }),
            'heads': self.num_heads
        })

    def set_attn_processor(self, processors):
        """Set attention processors."""
        for name, processor in processors.items():
            self.attn_processors[name] = processor

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                guidance=None, pooled_projections=None, return_dict=True, **kwargs):
        """Mock forward pass."""
        # Simulate transformer processing
        output = hidden_states + torch.randn_like(hidden_states) * 0.1

        if return_dict:
            return {'sample': output}
        return output


def create_mock_bezier_data() -> Dict[str, Any]:
    """Create mock Bézier curve data for demonstration."""
    return {
        'image_path': 'mock_calligraphy.jpg',
        'characters': [
            {
                'character_id': 0,
                'contour_area': 1250.0,
                'bounding_box': [50, 30, 80, 100],
                'bezier_curves': [
                    [[50, 30], [60, 40], [70, 50], [80, 60]],
                    [[80, 60], [90, 70], [100, 80], [110, 90]],
                    [[110, 90], [120, 100], [130, 110], [140, 120]]
                ],
                'original_contour_points': 150
            },
            {
                'character_id': 1,
                'contour_area': 980.0,
                'bounding_box': [150, 40, 70, 90],
                'bezier_curves': [
                    [[150, 40], [160, 50], [170, 60], [180, 70]],
                    [[180, 70], [190, 80], [200, 90], [210, 100]]
                ],
                'original_contour_points': 120
            }
        ]
    }


def demonstrate_spatial_attention_fuser():
    """Demonstrate SpatialAttentionFuser functionality."""

    print("=== SpatialAttentionFuser Demonstration ===\n")

    # 1. Initialize SpatialAttentionFuser
    print("1. Initializing SpatialAttentionFuser...")
    spatial_fuser = SpatialAttentionFuser(
        hidden_dim=3072,
        num_heads=24,
        head_dim=128,
        density_feature_dim=256,
        spatial_resolution=64,
        fusion_layers=3,
        use_positional_encoding=True,
        dropout=0.1,
        device='cpu',
        dtype=torch.float32
    )

    # 2. Check parameter count
    param_count = spatial_fuser.get_parameter_count()
    print(f"   Parameter count breakdown:")
    for component, count in param_count.items():
        print(f"     - {component}: {count:,}")

    target_params = 3800000  # 3.8M
    efficiency = param_count['total'] / target_params
    print(f"   Total parameters: {param_count['total']:,}")
    print(f"   Target parameters: {target_params:,}")
    print(f"   Parameter efficiency: {efficiency:.2f}")

    # 3. Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 2
    seq_len = 4096
    hidden_dim = 3072

    # Create test tensors
    attention_output = torch.randn(batch_size, seq_len, hidden_dim)
    density_map = torch.randn(batch_size, 1, 64, 64)

    # Forward pass
    fused_output, debug_info = spatial_fuser(
        attention_output=attention_output,
        density_map=density_map
    )

    print(f"   Input shape: {attention_output.shape}")
    print(f"   Density map shape: {density_map.shape}")
    print(f"   Output shape: {fused_output.shape}")
    print(f"   Debug info keys: {list(debug_info.keys())}")

    # 4. Validate output
    print("\n3. Validating output...")
    assert fused_output.shape == attention_output.shape
    assert 'density_features' in debug_info
    assert 'spatial_gates' in debug_info
    print("   ✓ Output validation passed")

    return spatial_fuser


def demonstrate_bezier_to_spatial_pipeline():
    """Demonstrate the complete Bézier → Spatial pipeline."""

    print("\n=== Bézier → Spatial Pipeline Demonstration ===\n")

    # 1. Initialize BezierParameterProcessor
    print("1. Initializing BezierParameterProcessor...")
    bezier_processor = BezierParameterProcessor(
        output_size=(64, 64),
        hidden_dim=256,
        device='cpu'
    )

    # 2. Create mock Bézier data
    print("2. Creating mock Bézier data...")
    bezier_data = create_mock_bezier_data()
    print(f"   Characters: {len(bezier_data['characters'])}")
    print(f"   Total curves: {sum(len(char['bezier_curves']) for char in bezier_data['characters'])}")

    # 3. Process Bézier data to density map
    print("3. Processing Bézier data to density map...")
    density_map = bezier_processor.process_bezier_data(bezier_data)
    print(f"   Density map shape: {density_map.shape}")
    print(f"   Density range: [{density_map.min():.4f}, {density_map.max():.4f}]")

    # 4. Initialize SpatialAttentionFuser
    print("4. Initializing SpatialAttentionFuser...")
    spatial_fuser = SpatialAttentionFuser(
        hidden_dim=3072,
        device='cpu',
        dtype=torch.float32
    )

    # 5. Test complete pipeline
    print("5. Testing complete pipeline...")
    batch_size = density_map.shape[0]
    attention_output = torch.randn(batch_size, 4096, 3072)

    # Apply spatial attention fusion
    fused_output, debug_info = spatial_fuser(
        attention_output=attention_output,
        density_map=density_map
    )

    print(f"   Pipeline input: {attention_output.shape}")
    print(f"   Pipeline output: {fused_output.shape}")
    print(f"   Density features: {debug_info['density_features'].shape}")
    print(f"   Spatial gates: {debug_info['spatial_gates'].shape}")

    # 6. Validate pipeline
    print("6. Validating pipeline...")
    assert fused_output.shape == attention_output.shape
    assert debug_info['density_features'].shape[0] == batch_size
    print("   ✓ Pipeline validation passed")

    return bezier_processor, spatial_fuser, density_map


def demonstrate_flux_integration():
    """Demonstrate integration with FLUX transformer."""

    print("\n=== FLUX Transformer Integration Demonstration ===\n")

    # 1. Create mock FLUX transformer
    print("1. Creating mock FLUX transformer...")
    flux_transformer = MockFluxTransformer()
    print(f"   Double-stream blocks: {len(flux_transformer.transformer_blocks)}")
    print(f"   Single-stream blocks: {len(flux_transformer.single_transformer_blocks)}")

    # 2. Initialize SpatialTransformerIntegrator
    print("2. Initializing SpatialTransformerIntegrator...")
    spatial_integrator = SpatialTransformerIntegrator(
        flux_transformer=flux_transformer,
        device='cpu',
        dtype=torch.float32
    )

    # 3. Get integration summary before integration
    print("3. Getting integration summary (before integration)...")
    summary = spatial_integrator.get_integration_summary()
    print(f"   Total parameters: {summary['total_parameters']:,}")
    print(f"   Integration status: {summary['integration_status']}")
    print(f"   Phase 1 blocks: {summary['phase_breakdown']['phase1_double_stream']['num_blocks']}")
    print(f"   Phase 3 blocks: {summary['phase_breakdown']['phase3_single_stream']['num_blocks']}")

    # 4. Integrate spatial attention
    print("4. Integrating spatial attention...")
    spatial_integrator.integrate_spatial_attention()

    # 5. Get integration summary after integration
    print("5. Getting integration summary (after integration)...")
    summary = spatial_integrator.get_integration_summary()
    print(f"   Integration status: {summary['integration_status']}")
    print(f"   Parameter efficiency: {summary['parameter_efficiency']:.2f}")

    # 6. Validate integration
    print("6. Validating integration...")
    validation_results = validate_spatial_integration(spatial_integrator)
    print(f"   Integration valid: {validation_results['integration_status']}")
    print(f"   Parameter count OK: {validation_results['parameter_count_ok']}")
    print(f"   Parameter efficiency: {validation_results['parameter_efficiency']:.2f}")

    if validation_results['recommendations']:
        print("   Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"     - {rec}")
    else:
        print("   ✓ No recommendations - integration is optimal")

    # 7. Test forward pass with spatial conditioning
    print("7. Testing forward pass with spatial conditioning...")
    batch_size = 1
    seq_len = 4096
    hidden_dim = 3072

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    encoder_hidden_states = torch.randn(batch_size, 256, hidden_dim)
    timestep = torch.tensor([500])
    density_map = torch.randn(batch_size, 1, 64, 64)

    # Forward pass
    output = spatial_integrator.forward_with_spatial_conditioning(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        density_map=density_map,
        return_dict=True
    )

    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Output shape: {output['sample'].shape}")
    print("   ✓ Forward pass successful")

    # 8. Cleanup
    print("8. Cleanup...")
    spatial_integrator.remove_spatial_attention()
    print("   ✓ Spatial attention removed")

    return spatial_integrator


def demonstrate_complete_bezier_spatial_pipeline():
    """Demonstrate the complete BezierSpatialPipeline."""

    print("\n=== Complete BezierSpatialPipeline Demonstration ===\n")

    # 1. Initialize components
    print("1. Initializing components...")
    flux_transformer = MockFluxTransformer()
    bezier_processor = BezierParameterProcessor(
        output_size=(64, 64),
        hidden_dim=256,
        device='cpu'
    )

    # 2. Create BezierSpatialPipeline
    print("2. Creating BezierSpatialPipeline...")
    pipeline = BezierSpatialPipeline(
        flux_transformer=flux_transformer,
        bezier_processor=bezier_processor
    )

    # 3. Setup spatial conditioning
    print("3. Setting up spatial conditioning...")
    pipeline.setup_spatial_conditioning()

    # 4. Get pipeline summary
    print("4. Getting pipeline summary...")
    summary = pipeline.get_pipeline_summary()
    print(f"   Bezier processor active: {summary['bezier_processor_active']}")
    print(f"   Pipeline ready: {summary['pipeline_ready']}")
    print(f"   Total parameters: {summary['spatial_attention_integration']['total_parameters']:,}")

    # 5. Test generation with Bézier conditioning
    print("5. Testing generation with Bézier conditioning...")
    bezier_data = create_mock_bezier_data()

    batch_size = 1
    seq_len = 4096
    hidden_dim = 3072

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    encoder_hidden_states = torch.randn(batch_size, 256, hidden_dim)
    timestep = torch.tensor([500])

    # Generate with Bézier conditioning
    output = pipeline.generate_with_bezier_conditioning(
        bezier_data=bezier_data,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        return_dict=True
    )

    print(f"   Generated output shape: {output['sample'].shape}")
    print("   ✓ Generation with Bézier conditioning successful")

    # 6. Cleanup
    print("6. Cleanup...")
    pipeline.cleanup()
    print("   ✓ Pipeline cleanup complete")

    return pipeline


def demonstrate_performance_characteristics():
    """Demonstrate performance characteristics of SpatialAttentionFuser."""

    print("\n=== Performance Characteristics Demonstration ===\n")

    # 1. Initialize components
    print("1. Initializing components for performance testing...")
    spatial_fuser = SpatialAttentionFuser(
        hidden_dim=3072,
        device='cpu',
        dtype=torch.float32
    )

    # 2. Test different batch sizes
    print("2. Testing different batch sizes...")
    batch_sizes = [1, 2, 4, 8]
    seq_len = 4096
    hidden_dim = 3072

    for batch_size in batch_sizes:
        attention_output = torch.randn(batch_size, seq_len, hidden_dim)
        density_map = torch.randn(batch_size, 1, 64, 64)

        # Time the forward pass
        import time
        start_time = time.time()

        with torch.no_grad():
            fused_output, debug_info = spatial_fuser(
                attention_output=attention_output,
                density_map=density_map
            )

        end_time = time.time()
        processing_time = end_time - start_time

        print(f"   Batch size {batch_size}: {processing_time:.4f}s")

    # 3. Test memory usage
    print("3. Testing memory usage...")

    # Before processing
    initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

    # Process large batch
    large_batch_size = 4
    attention_output = torch.randn(large_batch_size, seq_len, hidden_dim)
    density_map = torch.randn(large_batch_size, 1, 64, 64)

    with torch.no_grad():
        fused_output, debug_info = spatial_fuser(
            attention_output=attention_output,
            density_map=density_map
        )

    # After processing
    final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    memory_used = final_memory - initial_memory

    if torch.cuda.is_available():
        print(f"   Memory used: {memory_used / 1e6:.2f} MB")
    else:
        print(f"   Memory tracking not available on CPU")

    # 4. Test gradient computation
    print("4. Testing gradient computation...")
    attention_output = torch.randn(2, seq_len, hidden_dim, requires_grad=True)
    density_map = torch.randn(2, 1, 64, 64, requires_grad=True)

    fused_output, debug_info = spatial_fuser(
        attention_output=attention_output,
        density_map=density_map
    )

    # Compute gradients
    loss = fused_output.sum()
    loss.backward()

    # Check gradients
    attention_grad_norm = attention_output.grad.norm().item()
    density_grad_norm = density_map.grad.norm().item()

    print(f"   Attention gradient norm: {attention_grad_norm:.4f}")
    print(f"   Density gradient norm: {density_grad_norm:.4f}")
    print("   ✓ Gradient computation successful")


def visualize_architecture_integration():
    """Visualize the architecture integration mapping."""

    print("\n=== Architecture Integration Visualization ===\n")

    # Architecture mapping from TASK_01
    integration_phases = {
        'phase1_double_stream': list(range(12, 19)),  # blocks 12-18
        'phase3_single_stream': list(range(20, 38))   # blocks 20-37
    }

    print("FLUX Transformer Architecture Integration:")
    print("="*50)

    # Double-stream blocks
    print("Double-Stream Blocks (0-18):")
    for i in range(19):
        if i in integration_phases['phase1_double_stream']:
            print(f"  Block {i:2d}: [ENHANCED] ← SpatialAttentionFuser")
        else:
            print(f"  Block {i:2d}: [STANDARD]")

    print()

    # Single-stream blocks
    print("Single-Stream Blocks (19-56):")
    for i in range(19, 57):
        if i in integration_phases['phase3_single_stream']:
            print(f"  Block {i:2d}: [ENHANCED] ← SpatialAttentionFuser")
        else:
            print(f"  Block {i:2d}: [STANDARD]")

    print()
    print("Integration Summary:")
    print(f"  - Phase 1 (Double-Stream): {len(integration_phases['phase1_double_stream'])} blocks")
    print(f"  - Phase 3 (Single-Stream): {len(integration_phases['phase3_single_stream'])} blocks")
    print(f"  - Total enhanced blocks: {len(integration_phases['phase1_double_stream']) + len(integration_phases['phase3_single_stream'])}")
    print(f"  - Total standard blocks: {57 - (len(integration_phases['phase1_double_stream']) + len(integration_phases['phase3_single_stream']))}")


def main():
    """Main demonstration function."""

    print("SpatialAttentionFuser Integration Example")
    print("="*50)
    print("This example demonstrates the complete integration of")
    print("SpatialAttentionFuser with FLUX transformer for BezierAdapter.")
    print()

    try:
        # 1. Basic SpatialAttentionFuser demonstration
        spatial_fuser = demonstrate_spatial_attention_fuser()

        # 2. Bézier to Spatial pipeline demonstration
        bezier_processor, spatial_fuser, density_map = demonstrate_bezier_to_spatial_pipeline()

        # 3. FLUX integration demonstration
        spatial_integrator = demonstrate_flux_integration()

        # 4. Complete pipeline demonstration
        pipeline = demonstrate_complete_bezier_spatial_pipeline()

        # 5. Performance characteristics
        demonstrate_performance_characteristics()

        # 6. Architecture visualization
        visualize_architecture_integration()

        print("\n" + "="*50)
        print("✓ All demonstrations completed successfully!")
        print("SpatialAttentionFuser integration is ready for TASK_05.")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run the main demonstration
    main()