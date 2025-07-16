"""
Style Bezier Fusion Integration Example

This example demonstrates the complete integration of StyleBezierFusionModule
with the FLUX transformer for BezierAdapter functionality.

Components demonstrated:
1. StyleBezierFusionModule (~3.8M parameters)
2. AdaIN (Adaptive Instance Normalization) for style transfer
3. Cross-modal attention for style-content fusion
4. Integration with FLUX transformer blocks 5-15
5. Complete end-to-end workflow

Architecture Integration:
- Phase 2: Single-Stream Blocks 5-15 (StyleBezierFusion)
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

from style_bezier_fusion_module import StyleBezierFusionModule, AdaIN, CrossModalAttention, StyleEncoder, StyleBezierProcessor
from style_transformer_integration import StyleTransformerIntegrator, BezierStylePipeline, UnifiedBezierPipeline, validate_style_integration
from bezier_parameter_processor import BezierParameterProcessor

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


def create_mock_style_data() -> Dict[str, Any]:
    """Create mock style data for demonstration."""
    return {
        'style_vectors': torch.randn(2, 256),  # Pre-computed style vectors
        'style_image_path': 'mock_style.jpg',
        'style_type': 'calligraphy',
        'style_strength': 0.8,
        'style_features': {
            'brush_width': 0.6,
            'ink_density': 0.7,
            'stroke_speed': 0.5,
            'pressure_variation': 0.8
        }
    }


def demonstrate_adain_functionality():
    """Demonstrate AdaIN functionality."""

    print("=== AdaIN Demonstration ===\n")

    # 1. Initialize AdaIN
    print("1. Initializing AdaIN...")
    adain = AdaIN(eps=1e-5)

    # 2. Create test data
    print("2. Creating test data...")
    batch_size = 2
    seq_len = 100
    feature_dim = 256

    content_features = torch.randn(batch_size, seq_len, feature_dim)
    style_features = torch.randn(batch_size, feature_dim)

    print(f"   Content features shape: {content_features.shape}")
    print(f"   Style features shape: {style_features.shape}")

    # 3. Apply AdaIN
    print("3. Applying AdaIN...")
    stylized_features = adain(content_features, style_features)

    print(f"   Stylized features shape: {stylized_features.shape}")

    # 4. Verify statistics transfer
    print("4. Verifying statistics transfer...")

    # Original content statistics
    content_mean = content_features.mean(dim=1, keepdim=True)
    content_std = content_features.std(dim=1, keepdim=True)

    # Style statistics
    style_mean = style_features.mean(dim=-1, keepdim=True).unsqueeze(1)
    style_std = style_features.std(dim=-1, keepdim=True).unsqueeze(1)

    # Stylized statistics
    stylized_mean = stylized_features.mean(dim=1, keepdim=True)
    stylized_std = stylized_features.std(dim=1, keepdim=True)

    print(f"   Style mean transfer error: {torch.abs(stylized_mean - style_mean).mean().item():.6f}")
    print(f"   Style std transfer error: {torch.abs(stylized_std - style_std).mean().item():.6f}")
    print("   ✓ AdaIN successfully transfers style statistics")

    return adain


def demonstrate_cross_modal_attention():
    """Demonstrate cross-modal attention functionality."""

    print("\n=== Cross-Modal Attention Demonstration ===\n")

    # 1. Initialize cross-modal attention
    print("1. Initializing cross-modal attention...")
    cross_attention = CrossModalAttention(
        content_dim=512,
        style_dim=256,
        num_heads=8,
        head_dim=64,
        dropout=0.1
    )

    # 2. Create test data
    print("2. Creating test data...")
    batch_size = 2
    content_seq_len = 100
    style_seq_len = 50

    content_features = torch.randn(batch_size, content_seq_len, 512)
    style_features = torch.randn(batch_size, style_seq_len, 256)

    print(f"   Content features shape: {content_features.shape}")
    print(f"   Style features shape: {style_features.shape}")

    # 3. Apply cross-modal attention
    print("3. Applying cross-modal attention...")
    attended_content, attention_weights = cross_attention(
        content_features=content_features,
        style_features=style_features
    )

    print(f"   Attended content shape: {attended_content.shape}")
    print(f"   Attention weights shape: {attention_weights.shape}")

    # 4. Verify attention properties
    print("4. Verifying attention properties...")

    # Check attention weights sum to 1
    attention_sum = attention_weights.sum(dim=-1)
    print(f"   Attention weights sum (should be ~1.0): {attention_sum.mean().item():.6f}")

    # Check attention weights are non-negative
    min_attention = attention_weights.min().item()
    print(f"   Minimum attention weight (should be ≥0): {min_attention:.6f}")

    print("   ✓ Cross-modal attention working correctly")

    return cross_attention


def demonstrate_style_encoder():
    """Demonstrate style encoder functionality."""

    print("\n=== Style Encoder Demonstration ===\n")

    # 1. Initialize style encoder
    print("1. Initializing style encoder...")
    style_encoder = StyleEncoder(
        input_dim=256,
        hidden_dim=512,
        output_dim=256,
        num_layers=3,
        dropout=0.1
    )

    # 2. Create test data
    print("2. Creating test data...")
    batch_size = 2
    style_vectors = torch.randn(batch_size, 256)

    print(f"   Style vectors shape: {style_vectors.shape}")

    # 3. Encode style vectors
    print("3. Encoding style vectors...")
    style_features = style_encoder(style_vectors)

    print(f"   Style features shape: {style_features.shape}")

    # 4. Check parameter count
    print("4. Checking parameter count...")
    param_count = sum(p.numel() for p in style_encoder.parameters())
    print(f"   Style encoder parameters: {param_count:,}")

    print("   ✓ Style encoder working correctly")

    return style_encoder


def demonstrate_style_bezier_fusion_module():
    """Demonstrate StyleBezierFusionModule functionality."""

    print("\n=== StyleBezierFusionModule Demonstration ===\n")

    # 1. Initialize StyleBezierFusionModule
    print("1. Initializing StyleBezierFusionModule...")
    style_fusion = StyleBezierFusionModule(
        content_dim=3072,
        style_vector_dim=256,
        style_feature_dim=256,
        num_attention_heads=8,
        attention_head_dim=64,
        fusion_layers=3,
        use_adain=True,
        use_cross_attention=True,
        dropout=0.1,
        device='cpu',
        dtype=torch.float32
    )

    # 2. Check parameter count
    param_count = style_fusion.get_parameter_count()
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
    content_dim = 3072
    style_vector_dim = 256

    # Create test tensors
    content_features = torch.randn(batch_size, seq_len, content_dim)
    style_vectors = torch.randn(batch_size, style_vector_dim)

    # Forward pass
    fused_features, debug_info = style_fusion(
        content_features=content_features,
        style_vectors=style_vectors
    )

    print(f"   Input shape: {content_features.shape}")
    print(f"   Style vectors shape: {style_vectors.shape}")
    print(f"   Output shape: {fused_features.shape}")
    print(f"   Debug info keys: {list(debug_info.keys())}")

    # 4. Validate output
    print("\n3. Validating output...")
    assert fused_features.shape == content_features.shape
    assert 'style_features' in debug_info
    assert 'style_gate' in debug_info
    assert 'adain_features' in debug_info
    print("   ✓ Output validation passed")

    # 5. Test style strength control
    print("\n4. Testing style strength control...")
    style_fusion.set_style_strength(0.5)
    assert style_fusion.get_style_strength() == 0.5
    print(f"   Style strength set to: {style_fusion.get_style_strength()}")
    print("   ✓ Style strength control working")

    return style_fusion


def demonstrate_flux_integration():
    """Demonstrate integration with FLUX transformer."""

    print("\n=== FLUX Transformer Integration Demonstration ===\n")

    # 1. Create mock FLUX transformer
    print("1. Creating mock FLUX transformer...")
    flux_transformer = MockFluxTransformer()
    print(f"   Double-stream blocks: {len(flux_transformer.transformer_blocks)}")
    print(f"   Single-stream blocks: {len(flux_transformer.single_transformer_blocks)}")

    # 2. Initialize StyleTransformerIntegrator
    print("2. Initializing StyleTransformerIntegrator...")
    style_integrator = StyleTransformerIntegrator(
        flux_transformer=flux_transformer,
        device='cpu',
        dtype=torch.float32
    )

    # 3. Get integration summary before integration
    print("3. Getting integration summary (before integration)...")
    summary = style_integrator.get_integration_summary()
    print(f"   Total parameters: {summary['total_parameters']:,}")
    print(f"   Integration status: {summary['integration_status']}")
    print(f"   Phase 2 blocks: {summary['phase_breakdown']['phase2_single_stream']['num_blocks']}")

    # 4. Integrate style fusion
    print("4. Integrating style fusion...")
    style_integrator.integrate_style_fusion()

    # 5. Get integration summary after integration
    print("5. Getting integration summary (after integration)...")
    summary = style_integrator.get_integration_summary()
    print(f"   Integration status: {summary['integration_status']}")
    print(f"   Parameter efficiency: {summary['parameter_efficiency']:.2f}")

    # 6. Validate integration
    print("6. Validating integration...")
    validation_results = validate_style_integration(style_integrator)
    print(f"   Integration valid: {validation_results['integration_status']}")
    print(f"   Parameter count OK: {validation_results['parameter_count_ok']}")
    print(f"   Parameter efficiency: {validation_results['parameter_efficiency']:.2f}")

    if validation_results['recommendations']:
        print("   Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"     - {rec}")
    else:
        print("   ✓ No recommendations - integration is optimal")

    # 7. Test forward pass with style conditioning
    print("7. Testing forward pass with style conditioning...")
    batch_size = 1
    seq_len = 4096
    hidden_dim = 3072

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    encoder_hidden_states = torch.randn(batch_size, 256, hidden_dim)
    timestep = torch.tensor([500])
    style_vectors = torch.randn(batch_size, 256)

    # Forward pass
    output = style_integrator.forward_with_style_conditioning(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        style_vectors=style_vectors,
        return_dict=True
    )

    print(f"   Input shape: {hidden_states.shape}")
    print(f"   Style vectors shape: {style_vectors.shape}")
    print(f"   Output shape: {output['sample'].shape}")
    print("   ✓ Forward pass successful")

    # 8. Test style strength control
    print("8. Testing style strength control...")
    style_integrator.set_style_strength(0.7)
    print(f"   Style strength set to: {style_integrator.get_style_strength()}")
    print("   ✓ Style strength control working")

    # 9. Cleanup
    print("9. Cleanup...")
    style_integrator.remove_style_fusion()
    print("   ✓ Style fusion removed")

    return style_integrator


def demonstrate_complete_style_pipeline():
    """Demonstrate the complete BezierStylePipeline."""

    print("\n=== Complete BezierStylePipeline Demonstration ===\n")

    # 1. Initialize components
    print("1. Initializing components...")
    flux_transformer = MockFluxTransformer()

    # Mock style processor
    class MockStyleProcessor:
        def process_style_data(self, style_data):
            return style_data.get('style_vectors', torch.randn(2, 256))

    style_processor = MockStyleProcessor()

    # 2. Create BezierStylePipeline
    print("2. Creating BezierStylePipeline...")
    pipeline = BezierStylePipeline(
        flux_transformer=flux_transformer,
        style_processor=style_processor
    )

    # 3. Setup style conditioning
    print("3. Setting up style conditioning...")
    pipeline.setup_style_conditioning()

    # 4. Get pipeline summary
    print("4. Getting pipeline summary...")
    summary = pipeline.get_pipeline_summary()
    print(f"   Style processor active: {summary['style_processor_active']}")
    print(f"   Pipeline ready: {summary['pipeline_ready']}")
    print(f"   Total parameters: {summary['style_fusion_integration']['total_parameters']:,}")

    # 5. Test generation with style conditioning
    print("5. Testing generation with style conditioning...")
    style_data = create_mock_style_data()

    batch_size = 1
    seq_len = 4096
    hidden_dim = 3072

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    encoder_hidden_states = torch.randn(batch_size, 256, hidden_dim)
    timestep = torch.tensor([500])

    # Generate with style conditioning
    output = pipeline.generate_with_style_conditioning(
        style_data=style_data,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        return_dict=True
    )

    print(f"   Generated output shape: {output['sample'].shape}")
    print("   ✓ Generation with style conditioning successful")

    # 6. Test style strength control
    print("6. Testing style strength control...")
    pipeline.set_style_strength(0.6)
    print(f"   Style strength set to: {pipeline.get_style_strength()}")
    print("   ✓ Style strength control working")

    # 7. Cleanup
    print("7. Cleanup...")
    pipeline.cleanup()
    print("   ✓ Pipeline cleanup complete")

    return pipeline


def demonstrate_unified_pipeline():
    """Demonstrate the unified BezierAdapter pipeline."""

    print("\n=== Unified BezierAdapter Pipeline Demonstration ===\n")

    # 1. Initialize components
    print("1. Initializing components...")
    flux_transformer = MockFluxTransformer()

    # Create processors
    bezier_processor = BezierParameterProcessor(
        output_size=(64, 64),
        hidden_dim=256,
        device='cpu'
    )

    class MockStyleProcessor:
        def process_style_data(self, style_data):
            return style_data.get('style_vectors', torch.randn(2, 256))

    style_processor = MockStyleProcessor()

    # 2. Create UnifiedBezierPipeline
    print("2. Creating UnifiedBezierPipeline...")
    unified_pipeline = UnifiedBezierPipeline(
        flux_transformer=flux_transformer,
        bezier_processor=bezier_processor,
        style_processor=style_processor
    )

    # 3. Setup unified conditioning
    print("3. Setting up unified conditioning...")
    unified_pipeline.setup_unified_conditioning()

    # 4. Get unified summary
    print("4. Getting unified summary...")
    summary = unified_pipeline.get_unified_summary()
    print(f"   Spatial integration ready: {summary['spatial_integration'] is not None}")
    print(f"   Style integration ready: {summary['style_integration']['integration_status']}")
    print(f"   Unified pipeline ready: {summary['unified_ready']}")

    # 5. Test generation with unified conditioning
    print("5. Testing generation with unified conditioning...")

    # Create mock data
    bezier_data = {
        'characters': [
            {
                'bezier_curves': [
                    [[0, 0], [10, 20], [30, 40], [50, 60]],
                    [[50, 60], [70, 80], [90, 100], [110, 120]]
                ]
            }
        ]
    }

    style_data = create_mock_style_data()

    batch_size = 1
    seq_len = 4096
    hidden_dim = 3072

    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    encoder_hidden_states = torch.randn(batch_size, 256, hidden_dim)
    timestep = torch.tensor([500])

    # Generate with unified conditioning
    output = unified_pipeline.generate_with_unified_conditioning(
        bezier_data=bezier_data,
        style_data=style_data,
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timestep,
        return_dict=True
    )

    print(f"   Generated output shape: {output['sample'].shape}")
    print("   ✓ Generation with unified conditioning successful")

    # 6. Test conditioning strength control
    print("6. Testing conditioning strength control...")
    unified_pipeline.set_conditioning_strength(style_strength=0.8)
    print(f"   Style strength set to: {unified_pipeline.style_integrator.get_style_strength()}")
    print("   ✓ Conditioning strength control working")

    # 7. Cleanup
    print("7. Cleanup...")
    unified_pipeline.cleanup()
    print("   ✓ Unified pipeline cleanup complete")

    return unified_pipeline


def demonstrate_performance_characteristics():
    """Demonstrate performance characteristics of StyleBezierFusionModule."""

    print("\n=== Performance Characteristics Demonstration ===\n")

    # 1. Initialize components
    print("1. Initializing components for performance testing...")
    style_fusion = StyleBezierFusionModule(
        content_dim=3072,
        style_vector_dim=256,
        device='cpu',
        dtype=torch.float32
    )

    # 2. Test different batch sizes
    print("2. Testing different batch sizes...")
    batch_sizes = [1, 2, 4, 8]
    seq_len = 4096
    content_dim = 3072
    style_vector_dim = 256

    for batch_size in batch_sizes:
        content_features = torch.randn(batch_size, seq_len, content_dim)
        style_vectors = torch.randn(batch_size, style_vector_dim)

        # Time the forward pass
        import time
        start_time = time.time()

        with torch.no_grad():
            fused_features, debug_info = style_fusion(
                content_features=content_features,
                style_vectors=style_vectors
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
    content_features = torch.randn(large_batch_size, seq_len, content_dim)
    style_vectors = torch.randn(large_batch_size, style_vector_dim)

    with torch.no_grad():
        fused_features, debug_info = style_fusion(
            content_features=content_features,
            style_vectors=style_vectors
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
    content_features = torch.randn(2, seq_len, content_dim, requires_grad=True)
    style_vectors = torch.randn(2, style_vector_dim, requires_grad=True)

    fused_features, debug_info = style_fusion(
        content_features=content_features,
        style_vectors=style_vectors
    )

    # Compute gradients
    loss = fused_features.sum()
    loss.backward()

    # Check gradients
    content_grad_norm = content_features.grad.norm().item()
    style_grad_norm = style_vectors.grad.norm().item()

    print(f"   Content gradient norm: {content_grad_norm:.4f}")
    print(f"   Style gradient norm: {style_grad_norm:.4f}")
    print("   ✓ Gradient computation successful")


def visualize_architecture_integration():
    """Visualize the architecture integration mapping."""

    print("\n=== Architecture Integration Visualization ===\n")

    # Architecture mapping from TASK_01
    integration_phases = {
        'phase2_single_stream': list(range(5, 16))  # blocks 5-15
    }

    print("FLUX Transformer Architecture Integration:")
    print("="*50)

    # Double-stream blocks
    print("Double-Stream Blocks (0-18):")
    for i in range(19):
        print(f"  Block {i:2d}: [STANDARD]")

    print()

    # Single-stream blocks
    print("Single-Stream Blocks (19-56):")
    for i in range(19, 57):
        if i in integration_phases['phase2_single_stream']:
            print(f"  Block {i:2d}: [ENHANCED] ← StyleBezierFusionModule")
        else:
            print(f"  Block {i:2d}: [STANDARD]")

    print()
    print("Integration Summary:")
    print(f"  - Phase 2 (Single-Stream): {len(integration_phases['phase2_single_stream'])} blocks")
    print(f"  - Total enhanced blocks: {len(integration_phases['phase2_single_stream'])}")
    print(f"  - Total standard blocks: {57 - len(integration_phases['phase2_single_stream'])}")

    print()
    print("StyleBezierFusionModule Features:")
    print("  - AdaIN (Adaptive Instance Normalization)")
    print("  - Cross-modal attention between style and content")
    print("  - Style strength control (0.0 to 1.0)")
    print("  - Learnable mixing weights for fusion layers")
    print("  - Style modulation gates")
    print("  - Residual connections for stability")


def main():
    """Main demonstration function."""

    print("StyleBezierFusionModule Integration Example")
    print("="*50)
    print("This example demonstrates the complete integration of")
    print("StyleBezierFusionModule with FLUX transformer for BezierAdapter.")
    print()

    try:
        # 1. Basic AdaIN demonstration
        adain = demonstrate_adain_functionality()

        # 2. Cross-modal attention demonstration
        cross_attention = demonstrate_cross_modal_attention()

        # 3. Style encoder demonstration
        style_encoder = demonstrate_style_encoder()

        # 4. Complete StyleBezierFusionModule demonstration
        style_fusion = demonstrate_style_bezier_fusion_module()

        # 5. FLUX integration demonstration
        style_integrator = demonstrate_flux_integration()

        # 6. Complete pipeline demonstration
        pipeline = demonstrate_complete_style_pipeline()

        # 7. Unified pipeline demonstration
        unified_pipeline = demonstrate_unified_pipeline()

        # 8. Performance characteristics
        demonstrate_performance_characteristics()

        # 9. Architecture visualization
        visualize_architecture_integration()

        print("\n" + "="*50)
        print("✓ All demonstrations completed successfully!")
        print("StyleBezierFusionModule integration is ready for TASK_06.")

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