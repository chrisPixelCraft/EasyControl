import torch
import torch.nn as nn
import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.spatial_attention_fuser import SpatialAttentionFuser, DensityAttentionFusionLayer, SpatialAttentionProcessor
from src.spatial_transformer_integration import SpatialTransformerIntegrator, BezierSpatialPipeline, validate_spatial_integration
from src.bezier_parameter_processor import BezierParameterProcessor

class TestSpatialAttentionFuser:
    """Test suite for SpatialAttentionFuser functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'  # Use CPU for testing
        self.batch_size = 2
        self.seq_len = 4096  # FLUX sequence length
        self.hidden_dim = 3072  # FLUX hidden dimension
        self.num_heads = 24
        self.head_dim = 128
        self.density_feature_dim = 256
        self.spatial_resolution = 64

        # Create SpatialAttentionFuser instance
        self.spatial_fuser = SpatialAttentionFuser(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            density_feature_dim=self.density_feature_dim,
            spatial_resolution=self.spatial_resolution,
            fusion_layers=3,
            use_positional_encoding=True,
            dropout=0.1,
            device=self.device,
            dtype=torch.float32
        )

        # Create test tensors
        self.attention_output = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        self.density_map = torch.randn(self.batch_size, 1, self.spatial_resolution, self.spatial_resolution)
        self.attention_weights = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.seq_len)

    def test_spatial_attention_fuser_initialization(self):
        """Test SpatialAttentionFuser initialization."""
        assert self.spatial_fuser.hidden_dim == self.hidden_dim
        assert self.spatial_fuser.num_heads == self.num_heads
        assert self.spatial_fuser.head_dim == self.head_dim
        assert self.spatial_fuser.density_feature_dim == self.density_feature_dim
        assert self.spatial_fuser.fusion_layers == 3

        # Check components exist
        assert hasattr(self.spatial_fuser, 'density_encoder')
        assert hasattr(self.spatial_fuser, 'spatial_pos_encoder')
        assert hasattr(self.spatial_fuser, 'fusion_layers_list')
        assert hasattr(self.spatial_fuser, 'spatial_attention_gates')
        assert hasattr(self.spatial_fuser, 'output_projection')

    def test_spatial_attention_fuser_forward(self):
        """Test SpatialAttentionFuser forward pass."""
        # Test forward pass
        fused_output, debug_info = self.spatial_fuser(
            attention_output=self.attention_output,
            density_map=self.density_map,
            attention_weights=self.attention_weights
        )

        # Check output shape
        assert fused_output.shape == self.attention_output.shape
        assert fused_output.shape == (self.batch_size, self.seq_len, self.hidden_dim)

        # Check debug info
        assert 'density_features' in debug_info
        assert 'spatial_gates' in debug_info
        assert 'layer_outputs' in debug_info
        assert 'combined_features' in debug_info

        # Check debug info shapes
        assert debug_info['density_features'].shape == (self.batch_size, self.density_feature_dim)
        assert debug_info['spatial_gates'].shape == (self.batch_size, self.num_heads, 1)
        assert len(debug_info['layer_outputs']) == 3  # fusion_layers
        assert debug_info['combined_features'].shape == (self.batch_size, self.seq_len, self.density_feature_dim)

    def test_parameter_count_target(self):
        """Test that parameter count meets the ~3.8M target."""
        param_count = self.spatial_fuser.get_parameter_count()

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
        assert param_count['density_encoder'] > 0
        assert param_count['spatial_pos_encoder'] > 0
        assert param_count['fusion_layers'] > 0
        assert param_count['spatial_attention_gates'] > 0
        assert param_count['output_projection'] > 0

    def test_spatial_position_generation(self):
        """Test spatial position generation."""
        positions = self.spatial_fuser._generate_spatial_positions(
            batch_size=self.batch_size,
            seq_len=self.seq_len
        )

        # Check shape
        assert positions.shape == (self.batch_size, self.seq_len, 2)

        # Check position range (should be normalized to [-1, 1])
        assert torch.all(positions >= -1)
        assert torch.all(positions <= 1)

    def test_gradient_flow(self):
        """Test gradient flow through the spatial attention fuser."""
        # Enable gradient tracking
        self.attention_output.requires_grad = True
        self.density_map.requires_grad = True

        # Forward pass
        fused_output, _ = self.spatial_fuser(
            attention_output=self.attention_output,
            density_map=self.density_map
        )

        # Backward pass
        loss = fused_output.sum()
        loss.backward()

        # Check gradients exist
        assert self.attention_output.grad is not None
        assert self.density_map.grad is not None

        # Check that model parameters have gradients
        for param in self.spatial_fuser.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDensityAttentionFusionLayer:
    """Test suite for DensityAttentionFusionLayer functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.seq_len = 4096
        self.hidden_dim = 3072
        self.num_heads = 24
        self.head_dim = 128
        self.density_feature_dim = 256

        # Create fusion layer
        self.fusion_layer = DensityAttentionFusionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            density_feature_dim=self.density_feature_dim,
            dropout=0.1
        )

        # Create test tensors
        self.attention_features = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        self.density_features = torch.randn(self.batch_size, self.seq_len, self.density_feature_dim)

    def test_fusion_layer_forward(self):
        """Test DensityAttentionFusionLayer forward pass."""
        output = self.fusion_layer(
            attention_features=self.attention_features,
            density_features=self.density_features
        )

        # Check output shape
        assert output.shape == self.attention_features.shape
        assert output.shape == (self.batch_size, self.seq_len, self.hidden_dim)

    def test_fusion_layer_with_mask(self):
        """Test DensityAttentionFusionLayer with spatial mask."""
        spatial_mask = torch.randint(0, 2, (self.batch_size, self.seq_len)).bool()

        output = self.fusion_layer(
            attention_features=self.attention_features,
            density_features=self.density_features,
            spatial_mask=spatial_mask
        )

        # Check output shape
        assert output.shape == self.attention_features.shape


class TestSpatialAttentionProcessor:
    """Test suite for SpatialAttentionProcessor functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'
        self.batch_size = 2
        self.seq_len = 4096
        self.hidden_dim = 3072

        # Create spatial attention fuser
        self.spatial_fuser = SpatialAttentionFuser(
            hidden_dim=self.hidden_dim,
            device=self.device,
            dtype=torch.float32
        )

        # Create spatial attention processor
        self.spatial_processor = SpatialAttentionProcessor(
            spatial_attention_fuser=self.spatial_fuser,
            base_processor=None  # Use default
        )

        # Mock attention module
        self.mock_attn = Mock()
        self.mock_attn.to_q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mock_attn.to_k = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mock_attn.to_v = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.mock_attn.to_out = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim)])
        self.mock_attn.heads = 24

        # Create test tensors
        self.hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim)
        self.density_map = torch.randn(self.batch_size, 1, 64, 64)

    def test_spatial_processor_without_density(self):
        """Test SpatialAttentionProcessor without density map."""
        output = self.spatial_processor(
            attn=self.mock_attn,
            hidden_states=self.hidden_states
        )

        # Check output shape
        assert output.shape == self.hidden_states.shape

    def test_spatial_processor_with_density(self):
        """Test SpatialAttentionProcessor with density map."""
        output = self.spatial_processor(
            attn=self.mock_attn,
            hidden_states=self.hidden_states,
            density_map=self.density_map
        )

        # Check output shape
        assert output.shape == self.hidden_states.shape


class TestSpatialTransformerIntegrator:
    """Test suite for SpatialTransformerIntegrator functionality."""

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

        # Create spatial transformer integrator
        self.integrator = SpatialTransformerIntegrator(
            flux_transformer=self.mock_flux_transformer,
            device=self.device,
            dtype=torch.float32
        )

    def test_integrator_initialization(self):
        """Test SpatialTransformerIntegrator initialization."""
        assert self.integrator.flux_transformer == self.mock_flux_transformer
        assert not self.integrator.is_integrated

        # Check phase configurations
        assert 'phase1_double_stream' in self.integrator.integration_phases
        assert 'phase3_single_stream' in self.integrator.integration_phases

        # Check spatial fusers are initialized
        assert len(self.integrator.phase1_spatial_fusers) == 7  # blocks 12-18
        assert len(self.integrator.phase3_spatial_fusers) == 18  # blocks 20-37

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
        assert 'phase1_double_stream' in summary['phase_breakdown']
        assert 'phase3_single_stream' in summary['phase_breakdown']

        # Check parameter counts
        assert summary['total_parameters'] > 0
        assert summary['target_parameters'] == 3800000

    def test_parameter_breakdown(self):
        """Test detailed parameter breakdown."""
        breakdown = self.integrator.get_detailed_parameter_breakdown()

        # Check structure
        assert 'phase1_double_stream' in breakdown
        assert 'phase3_single_stream' in breakdown

        # Check individual block breakdowns
        for phase in ['phase1_double_stream', 'phase3_single_stream']:
            for block_name, block_breakdown in breakdown[phase].items():
                assert 'total' in block_breakdown
                assert 'density_encoder' in block_breakdown
                assert 'fusion_layers' in block_breakdown
                assert block_breakdown['total'] > 0

    def test_integration_validation(self):
        """Test integration validation function."""
        validation_results = validate_spatial_integration(self.integrator)

        # Check validation structure
        assert 'integration_status' in validation_results
        assert 'parameter_count_ok' in validation_results
        assert 'parameter_efficiency' in validation_results
        assert 'phase_distribution' in validation_results
        assert 'detailed_breakdown' in validation_results
        assert 'recommendations' in validation_results

        # Check phase distribution
        assert validation_results['phase_distribution']['phase1_blocks'] == 7
        assert validation_results['phase_distribution']['phase3_blocks'] == 18


class TestBezierSpatialPipeline:
    """Test suite for BezierSpatialPipeline functionality."""

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

        # Create pipeline
        self.pipeline = BezierSpatialPipeline(
            flux_transformer=self.mock_flux_transformer,
            bezier_processor=self.mock_bezier_processor
        )

    def test_pipeline_initialization(self):
        """Test BezierSpatialPipeline initialization."""
        assert self.pipeline.flux_transformer == self.mock_flux_transformer
        assert self.pipeline.bezier_processor == self.mock_bezier_processor
        assert hasattr(self.pipeline, 'spatial_integrator')

    def test_pipeline_summary(self):
        """Test pipeline summary generation."""
        summary = self.pipeline.get_pipeline_summary()

        # Check summary structure
        assert 'spatial_attention_integration' in summary
        assert 'bezier_processor_active' in summary
        assert 'pipeline_ready' in summary

        # Check values
        assert summary['bezier_processor_active'] == True


class TestIntegrationWithBezierProcessor:
    """Test suite for integration with BezierParameterProcessor."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

        # Create real BezierParameterProcessor
        self.bezier_processor = BezierParameterProcessor(
            output_size=(64, 64),
            hidden_dim=256,
            device=self.device
        )

        # Create SpatialAttentionFuser
        self.spatial_fuser = SpatialAttentionFuser(
            hidden_dim=3072,
            device=self.device,
            dtype=torch.float32
        )

    def test_bezier_to_spatial_pipeline(self):
        """Test complete pipeline from Bézier data to spatial attention."""
        # Mock Bézier data (as would come from BezierCurveExtractor)
        mock_bezier_data = {
            'characters': [
                {
                    'bezier_curves': [
                        [[0, 0], [10, 20], [30, 40], [50, 60]],
                        [[50, 60], [70, 80], [90, 100], [110, 120]]
                    ]
                }
            ]
        }

        # Process Bézier data to density map
        density_map = self.bezier_processor.process_bezier_data(mock_bezier_data)

        # Check density map shape
        assert density_map.shape[1:] == (1, 64, 64)  # [B, 1, H, W]

        # Create mock attention output
        batch_size = density_map.shape[0]
        attention_output = torch.randn(batch_size, 4096, 3072)

        # Apply spatial attention fusion
        fused_output, debug_info = self.spatial_fuser(
            attention_output=attention_output,
            density_map=density_map
        )

        # Check output
        assert fused_output.shape == attention_output.shape
        assert 'density_features' in debug_info


def run_performance_benchmark():
    """Run performance benchmark for spatial attention integration."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create spatial attention fuser
    spatial_fuser = SpatialAttentionFuser(
        hidden_dim=3072,
        device=device,
        dtype=torch.float32
    )

    # Create test tensors
    batch_size = 1
    seq_len = 4096
    attention_output = torch.randn(batch_size, seq_len, 3072, device=device)
    density_map = torch.randn(batch_size, 1, 64, 64, device=device)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            fused_output, _ = spatial_fuser(attention_output, density_map)

    # Benchmark
    import time
    start_time = time.time()

    for _ in range(10):
        with torch.no_grad():
            fused_output, _ = spatial_fuser(attention_output, density_map)

    end_time = time.time()
    avg_time = (end_time - start_time) / 10

    print(f"Average forward pass time: {avg_time:.4f} seconds")
    print(f"Device: {device}")
    print(f"Memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB" if device == 'cuda' else "Memory usage: CPU")

    return avg_time


if __name__ == "__main__":
    # Run basic tests
    print("Running SpatialAttentionFuser tests...")

    # Test parameter counting
    print("\n=== Parameter Count Test ===")
    test_fuser = TestSpatialAttentionFuser()
    test_fuser.setup_method()
    test_fuser.test_parameter_count_target()

    # Test integration summary
    print("\n=== Integration Summary Test ===")
    test_integrator = TestSpatialTransformerIntegrator()
    test_integrator.setup_method()
    summary = test_integrator.integrator.get_integration_summary()

    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Target parameters: {summary['target_parameters']:,}")
    print(f"Parameter efficiency: {summary['parameter_efficiency']:.2f}")
    print(f"Phase 1 blocks: {summary['phase_breakdown']['phase1_double_stream']['num_blocks']}")
    print(f"Phase 3 blocks: {summary['phase_breakdown']['phase3_single_stream']['num_blocks']}")

    # Test validation
    print("\n=== Integration Validation Test ===")
    validation_results = validate_spatial_integration(test_integrator.integrator)
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

    print(f"\nSpatialAttentionFuser integration tests completed successfully!")
    print(f"Average processing time: {avg_time:.4f} seconds per forward pass")

    # Test with pytest if available
    try:
        import pytest
        print("\n=== Running pytest ===")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\nPytest not available, skipping automated tests")