"""
Integration tests for BezierAdapter framework.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.bezier_adapter import (
    BezierParameterProcessor,
    ConditionInjectionAdapter,
    SpatialAttentionFuser,
    DensityAdaptiveSampler,
    StyleBezierFusionModule,
    BezierCurve,
    BezierConfig
)


class TestBezierAdapterIntegration:
    """Integration tests for the complete BezierAdapter framework."""
    
    @pytest.fixture
    def sample_bezier_curves(self):
        """Create sample BezierCurve objects."""
        curves = [
            BezierCurve(
                control_points=torch.tensor([
                    [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]
                ], dtype=torch.float32),
                curve_id="diagonal",
                weight=1.0
            ),
            BezierCurve(
                control_points=torch.tensor([
                    [-1.0, 0.0], [-0.5, 0.5], [0.5, 0.5], [1.0, 0.0]
                ], dtype=torch.float32),
                curve_id="arch",
                weight=0.8
            )
        ]
        return curves
    
    @pytest.fixture
    def device(self):
        """Get available device."""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def test_parameter_processor_integration(self, sample_bezier_curves, device):
        """Test BezierParameterProcessor integration."""
        from src.bezier_adapter.data_utils import batch_bezier_curves
        
        # Setup processor
        processor = BezierParameterProcessor().to(device)
        
        # Prepare data
        bezier_points, bezier_mask = batch_bezier_curves(sample_bezier_curves)
        bezier_points = bezier_points.to(device)
        
        # Process
        density_map, field_map = processor(bezier_points)
        
        # Verify outputs
        assert density_map.device.type == device
        assert field_map.device.type == device
        assert density_map.shape == (1, 1, 256, 256)  # Default resolution
        assert field_map.shape == (1, 2, 256, 256)
        
        print(f"✅ BezierParameterProcessor integration test passed")
    
    def test_condition_adapter_integration(self, device):
        """Test ConditionInjectionAdapter integration."""
        adapter = ConditionInjectionAdapter().to(device)
        
        # Mock inputs
        style_features = torch.randn(1, 77, 768, device=device)
        bezier_coords = torch.randn(1, 8, 3, device=device)
        bezier_density = torch.rand(1, 8, device=device)
        
        # Process
        output = adapter(
            style_features=style_features,
            bezier_coords=bezier_coords,
            bezier_density=bezier_density
        )
        
        # Verify output
        assert output.device.type == device
        assert output.shape[-1] == 3072  # Should match FluxTransformer inner_dim
        assert not torch.isnan(output).any()
        
        print(f"✅ ConditionInjectionAdapter integration test passed")
    
    def test_spatial_attention_fuser_integration(self, device):
        """Test SpatialAttentionFuser integration."""
        fuser = SpatialAttentionFuser().to(device)
        
        # Mock inputs
        spatial_features = torch.randn(1, 64*64, 768, device=device)  # [B, H*W, D]
        density_weights = torch.rand(1, 64, 64, 1, device=device)    # [B, H, W, 1]
        
        # Process
        fused_features, attention_weights = fuser(spatial_features, density_weights)
        
        # Verify outputs
        assert fused_features.device.type == device
        assert attention_weights.device.type == device
        assert fused_features.shape == (1, 64*64, 768)
        assert not torch.isnan(fused_features).any()
        
        print(f"✅ SpatialAttentionFuser integration test passed")
    
    def test_density_sampler_integration(self):
        """Test DensityAdaptiveSampler integration."""
        sampler = DensityAdaptiveSampler()
        
        # Mock density map
        density_map = torch.rand(1, 1, 64, 64)
        
        # Create sampling strategy
        strategy = sampler.create_adaptive_sampling_strategy(density_map)
        
        # Verify strategy
        assert 'density_analysis' in strategy
        assert 'sampling_strategy' in strategy
        assert 'sampling_schedules' in strategy
        
        # Check efficiency metrics
        efficiency_metrics = sampler.get_sampling_efficiency_metrics(strategy)
        assert 'total_steps' in efficiency_metrics
        assert 'efficiency_gain_percent' in efficiency_metrics
        
        print(f"✅ DensityAdaptiveSampler integration test passed")
    
    def test_style_fusion_integration(self, device):
        """Test StyleBezierFusionModule integration."""
        fusion_module = StyleBezierFusionModule().to(device)
        
        # Mock inputs
        spatial_features = torch.randn(1, 128, 3072, device=device)  # [B, N, D]
        style_features = torch.randn(1, 768, device=device)         # [B, style_dim]
        bezier_density = torch.rand(1, 1, 32, 32, device=device)    # [B, 1, H, W]
        
        # Process
        fused_features = fusion_module(
            spatial_features=spatial_features,
            style_features=style_features,
            bezier_density=bezier_density
        )
        
        # Verify output
        assert fused_features.device.type == device
        assert fused_features.shape == (1, 128, 3072)
        assert not torch.isnan(fused_features).any()
        
        print(f"✅ StyleBezierFusionModule integration test passed")
    
    def test_end_to_end_pipeline(self, sample_bezier_curves, device):
        """Test end-to-end pipeline integration."""
        from src.bezier_adapter.data_utils import batch_bezier_curves
        
        # Initialize all modules
        processor = BezierParameterProcessor(output_resolution=(32, 32)).to(device)
        condition_adapter = ConditionInjectionAdapter().to(device)
        spatial_fuser = SpatialAttentionFuser(feature_dim=768).to(device)
        style_fusion = StyleBezierFusionModule(feature_dim=768).to(device)
        density_sampler = DensityAdaptiveSampler()
        
        # Prepare bezier data
        bezier_points, bezier_mask = batch_bezier_curves(sample_bezier_curves)
        bezier_points = bezier_points.to(device)
        
        # Step 1: Process bezier curves to density maps
        density_map, field_map = processor(bezier_points)
        
        # Step 2: Create adaptive sampling strategy
        sampling_strategy = density_sampler.create_adaptive_sampling_strategy(density_map)
        
        # Step 3: Multi-modal condition injection
        style_features = torch.randn(1, 77, 768, device=device)
        bezier_coords = torch.randn(1, 8, 3, device=device)
        bezier_density_1d = torch.rand(1, 8, device=device)
        
        condition_output = condition_adapter(
            style_features=style_features,
            bezier_coords=bezier_coords,
            bezier_density=bezier_density_1d
        )
        
        # Step 4: Spatial attention fusion
        spatial_features = torch.randn(1, 32*32, 768, device=device)
        density_weights = density_map.squeeze(1).unsqueeze(-1)  # [B, H, W, 1]
        
        fused_spatial, attention_weights = spatial_fuser(spatial_features, density_weights)
        
        # Step 5: Style-bezier fusion
        final_features = style_fusion(
            spatial_features=fused_spatial,
            style_features=style_features.mean(dim=1),  # Pool style features
            bezier_density=density_map
        )
        
        # Verify end-to-end processing
        assert final_features.device.type == device
        assert final_features.shape == (1, 32*32, 768)
        assert not torch.isnan(final_features).any()
        
        # Verify sampling strategy is valid
        assert sampling_strategy['sampling_strategy']['total_steps'].sum() > 0
        
        print(f"✅ End-to-end pipeline integration test passed")
    
    def test_parameter_efficiency(self, device):
        """Test parameter efficiency of the framework."""
        modules = {
            'BezierParameterProcessor': BezierParameterProcessor(),
            'ConditionInjectionAdapter': ConditionInjectionAdapter(),
            'SpatialAttentionFuser': SpatialAttentionFuser(),
            'StyleBezierFusionModule': StyleBezierFusionModule()
        }
        
        total_params = 0
        for name, module in modules.items():
            param_count = sum(p.numel() for p in module.parameters())
            total_params += param_count
            print(f"{name}: {param_count / 1e6:.1f}M parameters")
        
        print(f"Total BezierAdapter parameters: {total_params / 1e6:.1f}M")
        
        # Should be around 24.6M parameters as specified
        assert 20e6 < total_params < 30e6, f"Total parameters {total_params} outside expected range (20-30M)"
        
        print(f"✅ Parameter efficiency test passed")
    
    def test_memory_efficiency(self, device):
        """Test memory efficiency during processing."""
        if device == "cpu":
            pytest.skip("Memory test only relevant for GPU")
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create large batch to test memory usage
        processor = BezierParameterProcessor(output_resolution=(128, 128)).to(device)
        large_bezier_points = torch.randn(4, 16, 4, 2, device=device)  # Large batch
        
        # Process
        density_map, field_map = processor(large_bezier_points)
        
        peak_memory = torch.cuda.memory_allocated()
        memory_used = (peak_memory - initial_memory) / 1024**3  # GB
        
        # Clean up
        del density_map, field_map, large_bezier_points
        torch.cuda.empty_cache()
        
        print(f"Memory used: {memory_used:.2f} GB")
        
        # Should use reasonable amount of memory
        assert memory_used < 2.0, f"Memory usage too high: {memory_used:.2f}GB"
        
        print(f"✅ Memory efficiency test passed")


def test_framework_compatibility():
    """Test framework compatibility with different PyTorch versions."""
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
    
    # Test basic tensor operations
    x = torch.randn(2, 3, 4)
    y = torch.softmax(x, dim=-1)
    assert y.shape == x.shape
    
    print("✅ Framework compatibility test passed")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running integration tests on {device}")
    
    # Create test instance
    test_suite = TestBezierAdapterIntegration()
    
    # Sample bezier curves
    sample_curves = [
        BezierCurve(
            control_points=torch.tensor([
                [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]
            ], dtype=torch.float32)
        )
    ]
    
    try:
        # Run basic integration tests
        test_suite.test_parameter_processor_integration(sample_curves, device)
        test_suite.test_condition_adapter_integration(device)
        test_suite.test_spatial_attention_fuser_integration(device)
        test_suite.test_density_sampler_integration()
        test_suite.test_style_fusion_integration(device)
        test_suite.test_parameter_efficiency(device)
        
        # Run end-to-end test
        print("\nRunning end-to-end integration test...")
        test_suite.test_end_to_end_pipeline(sample_curves, device)
        
        print("\n🎉 All integration tests passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()