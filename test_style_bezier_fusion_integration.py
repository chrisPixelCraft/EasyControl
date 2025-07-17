import torch
import torch.nn as nn
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.style_bezier_fusion_module import StyleBezierFusionModule, AdaIN, CrossModalAttention, StyleEncoder, StyleBezierProcessor
from src.style_transformer_integration import StyleTransformerIntegrator, BezierStylePipeline, UnifiedBezierPipeline, validate_style_integration
from src.bezier_parameter_processor import BezierParameterProcessor

class TestAdaIN:
    """Test suite for AdaIN functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.seq_len = 100
        self.feature_dim = 256

        # Create AdaIN instance
        self.adain = AdaIN(eps=1e-5)

        # Create test tensors
        self.content_features = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        self.style_features = torch.randn(self.batch_size, self.feature_dim)

    def test_adain_initialization(self):
        """Test AdaIN initialization."""
        assert self.adain.eps == 1e-5

    def test_adain_forward_2d_style(self):
        """Test AdaIN forward with 2D style features."""
        stylized = self.adain(self.content_features, self.style_features)

        # Check output shape
        assert stylized.shape == self.content_features.shape

        # Check that style statistics are applied
        style_mean = self.style_features.mean(dim=-1, keepdim=True).unsqueeze(1)
        style_std = self.style_features.std(dim=-1, keepdim=True).unsqueeze(1)

        stylized_mean = stylized.mean(dim=1, keepdim=True)
        stylized_std = stylized.std(dim=1, keepdim=True)

        # Should be approximately equal (within tolerance)
        assert torch.allclose(stylized_mean, style_mean, atol=1e-3)
        assert torch.allclose(stylized_std, style_std, atol=1e-3)

    def test_adain_forward_3d_style(self):
        """Test AdaIN forward with 3D style features."""
        style_features_3d = torch.randn(self.batch_size, self.seq_len, self.feature_dim)
        stylized = self.adain(self.content_features, style_features_3d)

        # Check output shape
        assert stylized.shape == self.content_features.shape

    def test_adain_gradient_flow(self):
        """Test gradient flow through AdaIN."""
        content_features = torch.randn(self.batch_size, self.seq_len, self.feature_dim, requires_grad=True)
        style_features = torch.randn(self.batch_size, self.feature_dim, requires_grad=True)

        stylized = self.adain(content_features, style_features)
        loss = stylized.sum()
        loss.backward()

        # Check gradients exist
        assert content_features.grad is not None
        assert style_features.grad is not None


class TestCrossModalAttention:
    """Test suite for CrossModalAttention functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.content_seq_len = 100
        self.style_seq_len = 50
        self.content_dim = 512
        self.style_dim = 256
        self.num_heads = 8
        self.head_dim = 64

        # Create cross-modal attention instance
        self.cross_attention = CrossModalAttention(
            content_dim=self.content_dim,
            style_dim=self.style_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=0.1
        )

        # Create test tensors
        self.content_features = torch.randn(self.batch_size, self.content_seq_len, self.content_dim)
        self.style_features = torch.randn(self.batch_size, self.style_seq_len, self.style_dim)

    def test_cross_attention_initialization(self):
        """Test CrossModalAttention initialization."""
        assert self.cross_attention.content_dim == self.content_dim
        assert self.cross_attention.style_dim == self.style_dim
        assert self.cross_attention.num_heads == self.num_heads
        assert self.cross_attention.head_dim == self.head_dim

    def test_cross_attention_forward(self):
        """Test CrossModalAttention forward pass."""
        attended_content, attention_weights = self.cross_attention(
            content_features=self.content_features,
            style_features=self.style_features
        )

        # Check output shapes
        assert attended_content.shape == self.content_features.shape
        assert attention_weights.shape == (self.batch_size, self.content_seq_len, self.style_seq_len)

        # Check attention weights sum to 1
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones(self.batch_size, self.content_seq_len))

    def test_cross_attention_with_mask(self):
        """Test CrossModalAttention with attention mask."""
        attention_mask = torch.randint(0, 2, (self.batch_size, self.content_seq_len, self.style_seq_len)).bool()

        attended_content, attention_weights = self.cross_attention(
            content_features=self.content_features,
            style_features=self.style_features,
            attention_mask=attention_mask
        )

        # Check output shapes
        assert attended_content.shape == self.content_features.shape
        assert attention_weights.shape == (self.batch_size, self.content_seq_len, self.style_seq_len)

    def test_cross_attention_gradient_flow(self):
        """Test gradient flow through CrossModalAttention."""
        content_features = torch.randn(self.batch_size, self.content_seq_len, self.content_dim, requires_grad=True)
        style_features = torch.randn(self.batch_size, self.style_seq_len, self.style_dim, requires_grad=True)

        attended_content, attention_weights = self.cross_attention(
            content_features=content_features,
            style_features=style_features
        )

        loss = attended_content.sum()
        loss.backward()

        # Check gradients exist
        assert content_features.grad is not None
        assert style_features.grad is not None


class TestStyleEncoder:
    """Test suite for StyleEncoder functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.input_dim = 256
        self.hidden_dim = 512
        self.output_dim = 256
        self.num_layers = 3

        # Create style encoder instance
        self.style_encoder = StyleEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            dropout=0.1
        )

        # Create test tensors
        self.style_vectors = torch.randn(self.batch_size, self.input_dim)

    def test_style_encoder_initialization(self):
        """Test StyleEncoder initialization."""
        assert self.style_encoder.input_dim == self.input_dim
        assert self.style_encoder.hidden_dim == self.hidden_dim
        assert self.style_encoder.output_dim == self.output_dim
        assert self.style_encoder.num_layers == self.num_layers

    def test_style_encoder_forward(self):
        """Test StyleEncoder forward pass."""
        style_features = self.style_encoder(self.style_vectors)

        # Check output shape
        assert style_features.shape == (self.batch_size, self.output_dim)

    def test_style_encoder_gradient_flow(self):
        """Test gradient flow through StyleEncoder."""
        style_vectors = torch.randn(self.batch_size, self.input_dim, requires_grad=True)

        style_features = self.style_encoder(style_vectors)
        loss = style_features.sum()
        loss.backward()

        # Check gradients exist
        assert style_vectors.grad is not None


class TestStyleBezierFusionModule:
    """Test suite for StyleBezierFusionModule functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.seq_len = 4096  # FLUX sequence length
        self.content_dim = 3072  # FLUX hidden dimension
        self.style_vector_dim = 256
        self.style_feature_dim = 256

        # Create StyleBezierFusionModule instance
        self.style_fusion = StyleBezierFusionModule(
            content_dim=self.content_dim,
            style_vector_dim=self.style_vector_dim,
            style_feature_dim=self.style_feature_dim,
            num_attention_heads=8,
            attention_head_dim=64,
            fusion_layers=3,
            use_adain=True,
            use_cross_attention=True,
            dropout=0.1,
            device=self.device,
            dtype=torch.float32
        )

        # Create test tensors
        self.content_features = torch.randn(self.batch_size, self.seq_len, self.content_dim)
        self.style_vectors = torch.randn(self.batch_size, self.style_vector_dim)

    def test_style_fusion_initialization(self):
        """Test StyleBezierFusionModule initialization."""
        assert self.style_fusion.content_dim == self.content_dim
        assert self.style_fusion.style_vector_dim == self.style_vector_dim
        assert self.style_fusion.style_feature_dim == self.style_feature_dim
        assert self.style_fusion.fusion_layers == 3
        assert self.style_fusion.use_adain == True
        assert self.style_fusion.use_cross_attention == True

        # Check components exist
        assert hasattr(self.style_fusion, 'style_encoder')
        assert hasattr(self.style_fusion, 'adain')
        assert hasattr(self.style_fusion, 'cross_modal_attention_layers')
        assert hasattr(self.style_fusion, 'style_modulation')
        assert hasattr(self.style_fusion, 'fusion_network')
        assert hasattr(self.style_fusion, 'output_projection')

    def test_style_fusion_forward(self):
        """Test StyleBezierFusionModule forward pass."""
        fused_features, debug_info = self.style_fusion(
            content_features=self.content_features,
            style_vectors=self.style_vectors
        )

        # Check output shape
        assert fused_features.shape == self.content_features.shape

        # Check debug info
        assert 'style_features' in debug_info
        assert 'style_gate' in debug_info
        assert 'modulated_features' in debug_info
        assert 'fused_features' in debug_info
        assert 'mixing_weights' in debug_info
        assert 'attention_outputs' in debug_info
        assert 'attention_weights' in debug_info
        assert 'adain_features' in debug_info

        # Check debug info shapes
        assert debug_info['style_features'].shape == (self.batch_size, self.style_feature_dim)
        assert debug_info['style_gate'].shape == (self.batch_size, 1, self.content_dim)
        assert debug_info['modulated_features'].shape == self.content_features.shape
        assert debug_info['fused_features'].shape == self.content_features.shape
        assert debug_info['mixing_weights'].shape == (self.fusion_layers,)
        assert len(debug_info['attention_outputs']) == self.fusion_layers
        assert len(debug_info['attention_weights']) == self.fusion_layers

    def test_parameter_count_target(self):
        """Test that parameter count meets the ~3.8M target."""
        param_count = self.style_fusion.get_parameter_count()

        # Check total parameters
        total_params = param_count['total']
        target_params = 3800000  # 3.8M

        print(f"Total parameters: {total_params:,}")
        print(f"Target parameters: {target_params:,}")
        print(f"Parameter efficiency: {total_params / target_params:.2f}")

        # Should be within reasonable range of target (±20%)
        assert total_params <= target_params * 1.2, f"Parameter count {total_params} exceeds target {target_params} by more than 20%"
        assert total_params >= target_params * 0.8, f"Parameter count {total_params} is below target {target_params} by more than 20%"

        # Check individual component counts
        assert param_count['style_encoder'] > 0
        assert param_count['cross_modal_attention'] > 0
        assert param_count['style_modulation'] > 0
        assert param_count['fusion_network'] > 0
        assert param_count['output_projection'] > 0

    def test_style_fusion_without_adain(self):
        """Test StyleBezierFusionModule without AdaIN."""
        style_fusion_no_adain = StyleBezierFusionModule(
            content_dim=self.content_dim,
            style_vector_dim=self.style_vector_dim,
            use_adain=False,
            use_cross_attention=True,
            device=self.device
        )

        fused_features, debug_info = style_fusion_no_adain(
            content_features=self.content_features,
            style_vectors=self.style_vectors
        )

        # Check output shape
        assert fused_features.shape == self.content_features.shape

        # Check that AdaIN is not in debug info
        assert 'adain_features' not in debug_info

    def test_style_fusion_without_cross_attention(self):
        """Test StyleBezierFusionModule without cross-modal attention."""
        style_fusion_no_cross_attn = StyleBezierFusionModule(
            content_dim=self.content_dim,
            style_vector_dim=self.style_vector_dim,
            use_adain=True,
            use_cross_attention=False,
            device=self.device
        )

        fused_features, debug_info = style_fusion_no_cross_attn(
            content_features=self.content_features,
            style_vectors=self.style_vectors
        )

        # Check output shape
        assert fused_features.shape == self.content_features.shape

        # Check that cross-attention outputs are empty
        assert debug_info['attention_outputs'] == []
        assert debug_info['attention_weights'] == []

    def test_style_strength_control(self):
        """Test style strength control functionality."""
        # Test setting style strength
        self.style_fusion.set_style_strength(0.5)
        assert self.style_fusion.get_style_strength() == 0.5

        # Test clamping
        self.style_fusion.set_style_strength(1.5)
        assert self.style_fusion.get_style_strength() == 1.0

        self.style_fusion.set_style_strength(-0.5)
        assert self.style_fusion.get_style_strength() == 0.0

    def test_gradient_flow(self):
        """Test gradient flow through the style fusion module."""
        # Enable gradient tracking
        self.content_features.requires_grad = True
        self.style_vectors.requires_grad = True

        # Forward pass
        fused_features, debug_info = self.style_fusion(
            content_features=self.content_features,
            style_vectors=self.style_vectors
        )

        # Backward pass
        loss = fused_features.sum()
        loss.backward()

        # Check gradients exist
        assert self.content_features.grad is not None
        assert self.style_vectors.grad is not None

        # Check that model parameters have gradients
        for param in self.style_fusion.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestStyleBezierProcessor:
    """Test suite for StyleBezierProcessor functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.seq_len = 4096
        self.content_dim = 3072
        self.style_vector_dim = 256

        # Create style fusion module
        self.style_fusion = StyleBezierFusionModule(
            content_dim=self.content_dim,
            style_vector_dim=self.style_vector_dim,
            device=self.device
        )

        # Create style processor
        self.style_processor = StyleBezierProcessor(
            style_fusion_module=self.style_fusion,
            base_processor=None
        )

        # Mock attention module
        self.mock_attn = Mock()
        self.mock_attn.to_q = nn.Linear(self.content_dim, self.content_dim)
        self.mock_attn.to_k = nn.Linear(self.content_dim, self.content_dim)
        self.mock_attn.to_v = nn.Linear(self.content_dim, self.content_dim)
        self.mock_attn.to_out = nn.ModuleList([nn.Linear(self.content_dim, self.content_dim)])
        self.mock_attn.heads = 24

        # Create test tensors
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.content_dim)
        self.style_vectors = torch.randn(self.batch_size, self.style_vector_dim)

    def test_style_processor_without_style(self):
        """Test StyleBezierProcessor without style vectors."""
        output = self.style_processor(
            attn=self.mock_attn,
            hidden_states=self.hidden_states
        )

        # Check output shape
        assert output.shape == self.hidden_states.shape

    def test_style_processor_with_style(self):
        """Test StyleBezierProcessor with style vectors."""
        output = self.style_processor(
            attn=self.mock_attn,
            hidden_states=self.hidden_states,
            style_vectors=self.style_vectors
        )

        # Check output shape
        assert output.shape == self.hidden_states.shape


class TestStyleTransformerIntegrator:
    """Test suite for StyleTransformerIntegrator functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

        # Mock FLUX transformer
        self.mock_flux_transformer = Mock()
        self.mock_flux_transformer.transformer_blocks = [Mock() for _ in range(19)]
        self.mock_flux_transformer.single_transformer_blocks = [Mock() for _ in range(38)]
        self.mock_flux_transformer.attn_processors = {
            f'transformer_blocks.{i}.attn.processor': Mock() for i in range(19)
        }
        self.mock_flux_transformer.attn_processors.update({
            f'single_transformer_blocks.{i}.attn.processor': Mock() for i in range(38)
        })

        # Create style transformer integrator
        self.integrator = StyleTransformerIntegrator(
            flux_transformer=self.mock_flux_transformer,
            device=self.device,
            dtype=torch.float32
        )

    def test_integrator_initialization(self):
        """Test StyleTransformerIntegrator initialization."""
        assert self.integrator.flux_transformer == self.mock_flux_transformer
        assert not self.integrator.is_integrated

        # Check phase configurations
        assert 'phase2_single_stream' in self.integrator.integration_phases

        # Check style fusers are initialized
        assert len(self.integrator.phase2_style_fusers) == 11  # blocks 5-15

    def test_integration_summary(self):
        """Test integration summary generation."""
        summary = self.integrator.get_integration_summary()

        # Check summary structure
        assert 'total_parameters' in summary
        assert 'phase_breakdown' in summary
        assert 'integration_status' in summary
        assert 'target_parameters' in summary
        assert 'parameter_efficiency' in summary

        # Check phase breakdown
        assert 'phase2_single_stream' in summary['phase_breakdown']

        # Check parameter counts
        assert summary['total_parameters'] > 0
        assert summary['target_parameters'] == 3800000

    def test_parameter_breakdown(self):
        """Test detailed parameter breakdown."""
        breakdown = self.integrator.get_detailed_parameter_breakdown()

        # Check structure
        assert 'phase2_single_stream' in breakdown

        # Check individual block breakdowns
        for block_name, block_breakdown in breakdown['phase2_single_stream'].items():
            assert 'total' in block_breakdown
            assert 'style_encoder' in block_breakdown
            assert 'cross_modal_attention' in block_breakdown
            assert block_breakdown['total'] > 0

    def test_style_strength_control(self):
        """Test style strength control across all fusion modules."""
        # Test setting style strength
        self.integrator.set_style_strength(0.7)
        assert self.integrator.get_style_strength() == 0.7

        # Test all modules have the same strength
        for fuser in self.integrator.phase2_style_fusers.values():
            assert fuser.get_style_strength() == 0.7

    def test_integration_validation(self):
        """Test integration validation function."""
        validation_results = validate_style_integration(self.integrator)

        # Check validation structure
        assert 'integration_status' in validation_results
        assert 'parameter_count_ok' in validation_results
        assert 'parameter_efficiency' in validation_results
        assert 'phase_distribution' in validation_results
        assert 'detailed_breakdown' in validation_results
        assert 'recommendations' in validation_results

        # Check phase distribution
        assert validation_results['phase_distribution']['phase2_blocks'] == 11


class TestBezierStylePipeline:
    """Test suite for BezierStylePipeline functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

        # Mock components
        self.mock_flux_transformer = Mock()
        self.mock_flux_transformer.transformer_blocks = [Mock() for _ in range(19)]
        self.mock_flux_transformer.single_transformer_blocks = [Mock() for _ in range(38)]
        self.mock_flux_transformer.attn_processors = {}

        self.mock_style_processor = Mock()
        self.mock_style_processor.process_style_data = Mock(return_value=torch.randn(2, 256))

        # Create pipeline
        self.pipeline = BezierStylePipeline(
            flux_transformer=self.mock_flux_transformer,
            style_processor=self.mock_style_processor
        )

    def test_pipeline_initialization(self):
        """Test BezierStylePipeline initialization."""
        assert self.pipeline.flux_transformer == self.mock_flux_transformer
        assert self.pipeline.style_processor == self.mock_style_processor
        assert hasattr(self.pipeline, 'style_integrator')

    def test_pipeline_summary(self):
        """Test pipeline summary generation."""
        summary = self.pipeline.get_pipeline_summary()

        # Check summary structure
        assert 'style_fusion_integration' in summary
        assert 'style_processor_active' in summary
        assert 'pipeline_ready' in summary

        # Check values
        assert summary['style_processor_active'] == True

    def test_style_strength_control(self):
        """Test style strength control in pipeline."""
        self.pipeline.set_style_strength(0.8)
        assert self.pipeline.get_style_strength() == 0.8


class TestUnifiedBezierPipeline:
    """Test suite for UnifiedBezierPipeline functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

        # Mock components
        self.mock_flux_transformer = Mock()
        self.mock_flux_transformer.transformer_blocks = [Mock() for _ in range(19)]
        self.mock_flux_transformer.single_transformer_blocks = [Mock() for _ in range(38)]
        self.mock_flux_transformer.attn_processors = {}

        self.mock_bezier_processor = Mock()
        self.mock_bezier_processor.process_bezier_data = Mock(return_value=torch.randn(2, 1, 64, 64))

        self.mock_style_processor = Mock()
        self.mock_style_processor.process_style_data = Mock(return_value=torch.randn(2, 256))

        # Create unified pipeline
        self.unified_pipeline = UnifiedBezierPipeline(
            flux_transformer=self.mock_flux_transformer,
            bezier_processor=self.mock_bezier_processor,
            style_processor=self.mock_style_processor
        )

    def test_unified_pipeline_initialization(self):
        """Test UnifiedBezierPipeline initialization."""
        assert self.unified_pipeline.flux_transformer == self.mock_flux_transformer
        assert self.unified_pipeline.bezier_processor == self.mock_bezier_processor
        assert self.unified_pipeline.style_processor == self.mock_style_processor
        assert hasattr(self.unified_pipeline, 'style_integrator')

    def test_unified_summary(self):
        """Test unified pipeline summary generation."""
        summary = self.unified_pipeline.get_unified_summary()

        # Check summary structure
        assert 'spatial_integration' in summary
        assert 'style_integration' in summary
        assert 'unified_ready' in summary

    def test_conditioning_strength_control(self):
        """Test conditioning strength control in unified pipeline."""
        self.unified_pipeline.set_conditioning_strength(style_strength=0.6)
        assert self.unified_pipeline.style_integrator.get_style_strength() == 0.6


def run_performance_benchmark():
    """Run performance benchmark for style fusion integration."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create style fusion module
    style_fusion = StyleBezierFusionModule(
        content_dim=3072,
        style_vector_dim=256,
        device=device,
        dtype=torch.float32
    )

    # Create test tensors
    batch_size = 1
    seq_len = 4096
    content_features = torch.randn(batch_size, seq_len, 3072, device=device)
    style_vectors = torch.randn(batch_size, 256, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            fused_features, debug_info = style_fusion(content_features, style_vectors)

    # Benchmark
    import time
    start_time = time.time()

    for _ in range(10):
        with torch.no_grad():
            fused_features, debug_info = style_fusion(content_features, style_vectors)

    end_time = time.time()
    avg_time = (end_time - start_time) / 10

    print(f"Average forward pass time: {avg_time:.4f} seconds")
    print(f"Device: {device}")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if device == 'cuda' else "Memory usage: CPU")

    return avg_time


if __name__ == "__main__":
    # Run basic tests
    print("Running StyleBezierFusionModule tests...")

    # Test parameter counting
    print("\n=== Parameter Count Test ===")
    test_fusion = TestStyleBezierFusionModule()
    test_fusion.setup_method()
    test_fusion.test_parameter_count_target()

    # Test integration summary
    print("\n=== Integration Summary Test ===")
    test_integrator = TestStyleTransformerIntegrator()
    test_integrator.setup_method()
    summary = test_integrator.integrator.get_integration_summary()

    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Target parameters: {summary['target_parameters']:,}")
    print(f"Parameter efficiency: {summary['parameter_efficiency']:.2f}")
    print(f"Phase 2 blocks: {summary['phase_breakdown']['phase2_single_stream']['num_blocks']}")

    # Test validation
    print("\n=== Integration Validation Test ===")
    validation_results = validate_style_integration(test_integrator.integrator)
    print(f"Integration status: {validation_results['integration_status']}")
    print(f"Parameter count OK: {validation_results['parameter_count_ok']}")
    print(f"Parameter efficiency: {validation_results['parameter_efficiency']:.2f}")

    if validation_results['recommendations']:
        print("Recommendations:")
        for rec in validation_results['recommendations']:
            print(f"  - {rec}")

    # Run performance benchmark
    print("\n=== Performance Benchmark ===")
    avg_time = run_performance_benchmark()

    print(f"\nStyleBezierFusionModule integration tests completed successfully!")
    print(f"Average processing time: {avg_time:.4f} seconds per forward pass")

    # Test with pytest if available
    try:
        import pytest
        print("\n=== Running pytest ===")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\nPytest not available, skipping automated tests")