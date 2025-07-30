"""
Tests for BezierParameterProcessor module.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.bezier_adapter.bezier_processor import BezierParameterProcessor
from src.bezier_adapter.utils import BezierCurve, BezierConfig


class TestBezierParameterProcessor:
    """Test suite for BezierParameterProcessor."""
    
    @pytest.fixture
    def processor(self):
        """Create a BezierParameterProcessor instance."""
        return BezierParameterProcessor(
            output_resolution=(64, 64),
            hidden_dim=128,
            max_curves=8,
            max_points_per_curve=4
        )
    
    @pytest.fixture
    def sample_bezier_points(self):
        """Create sample bezier control points."""
        # Create a simple diagonal curve
        return torch.tensor([[
            [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]  # Single curve
        ]], dtype=torch.float32)  # [B=1, C=1, P=4, 2]
    
    @pytest.fixture
    def sample_mask(self):
        """Create sample mask for bezier points."""
        return torch.ones(1, 1, 4, dtype=torch.bool)  # [B=1, C=1, P=4]
    
    def test_initialization(self, processor):
        """Test processor initialization."""
        assert processor.output_resolution == (64, 64)
        assert processor.hidden_dim == 128
        assert processor.max_curves == 8
        assert processor.max_points_per_curve == 4
        
        # Check layers exist
        assert hasattr(processor, 'point_encoder')
        assert hasattr(processor, 'kde_bandwidth')
        assert hasattr(processor, 'density_threshold')
        assert hasattr(processor, 'density_proj')
        assert hasattr(processor, 'field_proj')
    
    def test_parameter_count(self, processor):
        """Test parameter count is reasonable."""
        param_count = processor.get_parameter_count()
        # Should be around 2.1M parameters
        assert 1e6 < param_count < 3e6, f"Parameter count {param_count} outside expected range"
    
    def test_forward_basic(self, processor, sample_bezier_points):
        """Test basic forward pass."""
        density_map, field_map = processor(sample_bezier_points)
        
        # Check output shapes
        assert density_map.shape == (1, 1, 64, 64)  # [B, 1, H, W]
        assert field_map.shape == (1, 2, 64, 64)    # [B, 2, H, W]
        
        # Check value ranges
        assert density_map.min() >= 0.0
        assert density_map.max() <= 1.0
        assert field_map.min() >= -1.0
        assert field_map.max() <= 1.0
        
        # Check no NaN values
        assert not torch.isnan(density_map).any()
        assert not torch.isnan(field_map).any()
    
    def test_forward_with_mask(self, processor, sample_bezier_points, sample_mask):
        """Test forward pass with mask."""
        density_map, field_map = processor(sample_bezier_points, mask=sample_mask)
        
        # Should produce same results as without mask (since mask is all True)
        density_map_no_mask, field_map_no_mask = processor(sample_bezier_points)
        
        assert torch.allclose(density_map, density_map_no_mask, atol=1e-5)
        assert torch.allclose(field_map, field_map_no_mask, atol=1e-5)
    
    def test_empty_curves(self, processor):
        """Test handling of empty curve input."""
        # Empty input
        empty_points = torch.zeros(1, 0, 4, 2)  # [B=1, C=0, P=4, 2]
        
        density_map, field_map = processor(empty_points)
        
        # Should return zero maps
        assert density_map.shape == (1, 1, 64, 64)
        assert field_map.shape == (1, 2, 64, 64)
        assert torch.allclose(density_map, torch.zeros_like(density_map), atol=1e-5)
    
    def test_multiple_curves(self, processor):
        """Test processing multiple curves."""
        # Create two curves
        bezier_points = torch.tensor([[
            [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],  # Diagonal
            [[-1.0, 0.0], [-0.5, 0.5], [0.5, 0.5], [1.0, 0.0]]   # Arch
        ]], dtype=torch.float32)  # [B=1, C=2, P=4, 2]
        
        density_map, field_map = processor(bezier_points)
        
        # Should produce non-zero density for multiple curves
        assert density_map.sum() > 0
        assert field_map.abs().sum() > 0
    
    def test_batch_processing(self, processor, sample_bezier_points):
        """Test batch processing."""
        # Create batch of 2 identical samples
        batch_points = sample_bezier_points.expand(2, -1, -1, -1)  # [B=2, C=1, P=4, 2]
        
        density_map, field_map = processor(batch_points)
        
        # Check batch dimension
        assert density_map.shape == (2, 1, 64, 64)
        assert field_map.shape == (2, 2, 64, 64)
        
        # Both samples should be identical
        assert torch.allclose(density_map[0], density_map[1], atol=1e-5)
        assert torch.allclose(field_map[0], field_map[1], atol=1e-5)
    
    def test_gradient_flow(self, processor, sample_bezier_points):
        """Test that gradients flow through the model."""
        # Enable gradients
        sample_bezier_points.requires_grad_(True)
        
        density_map, field_map = processor(sample_bezier_points)
        
        # Compute loss and backpropagate
        loss = density_map.sum() + field_map.sum()
        loss.backward()
        
        # Check that gradients exist
        assert sample_bezier_points.grad is not None
        assert not torch.allclose(sample_bezier_points.grad, torch.zeros_like(sample_bezier_points.grad))
    
    def test_kde_parameters_learned(self, processor):
        """Test that KDE parameters are learnable."""
        initial_bandwidth = processor.kde_bandwidth.clone()
        initial_threshold = processor.density_threshold.clone()
        
        # Check parameters require grad
        assert processor.kde_bandwidth.requires_grad
        assert processor.density_threshold.requires_grad
        
        # Simulate training step
        optimizer = torch.optim.Adam([processor.kde_bandwidth, processor.density_threshold], lr=0.01)
        
        bezier_points = torch.randn(1, 2, 4, 2)
        density_map, _ = processor(bezier_points)
        loss = density_map.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Parameters should have changed
        assert not torch.allclose(processor.kde_bandwidth, initial_bandwidth)
        assert not torch.allclose(processor.density_threshold, initial_threshold)
    
    def test_device_compatibility(self, processor):
        """Test device compatibility."""
        if torch.cuda.is_available():
            # Move to GPU
            processor_gpu = processor.cuda()
            bezier_points = torch.randn(1, 2, 4, 2).cuda()
            
            density_map, field_map = processor_gpu(bezier_points)
            
            assert density_map.device.type == 'cuda'
            assert field_map.device.type == 'cuda'
    
    def test_different_resolutions(self):
        """Test different output resolutions."""
        resolutions = [(32, 32), (64, 64), (128, 128)]
        
        for resolution in resolutions:
            processor = BezierParameterProcessor(output_resolution=resolution)
            bezier_points = torch.randn(1, 2, 4, 2)
            
            density_map, field_map = processor(bezier_points)
            
            assert density_map.shape == (1, 1, resolution[0], resolution[1])
            assert field_map.shape == (1, 2, resolution[0], resolution[1])
    
    def test_extreme_values(self, processor):
        """Test handling of extreme coordinate values."""
        # Points outside [-1, 1] range
        extreme_points = torch.tensor([[
            [[-2.0, -2.0], [2.0, 2.0], [3.0, -3.0], [-3.0, 3.0]]
        ]], dtype=torch.float32)
        
        density_map, field_map = processor(extreme_points)
        
        # Should not crash and produce valid outputs
        assert density_map.shape == (1, 1, 64, 64)
        assert field_map.shape == (1, 2, 64, 64)
        assert not torch.isnan(density_map).any()
        assert not torch.isnan(field_map).any()
    
    def test_identical_points(self, processor):
        """Test handling of identical control points."""
        # All points are the same
        identical_points = torch.tensor([[
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ]], dtype=torch.float32)
        
        density_map, field_map = processor(identical_points)
        
        # Should handle gracefully
        assert density_map.shape == (1, 1, 64, 64)
        assert field_map.shape == (1, 2, 64, 64)
        assert not torch.isnan(density_map).any()
        assert not torch.isnan(field_map).any()


def test_real_bezier_data():
    """Test with real bezier data format if available."""
    # This test uses actual data from bezier extraction if available
    dataset_path = "bezier_curves_output_no_visualization"
    
    if not Path(dataset_path).exists():
        pytest.skip("Real bezier dataset not available")
    
    try:
        from src.bezier_adapter.data_utils import load_sample_bezier_data
        
        bezier_points, bezier_mask = load_sample_bezier_data(dataset_path)
        
        if bezier_points.numel() == 0:
            pytest.skip("No bezier data found in dataset")
        
        processor = BezierParameterProcessor()
        density_map, field_map = processor(bezier_points)
        
        # Should process real data successfully
        assert density_map.shape[0] == bezier_points.shape[0]
        assert not torch.isnan(density_map).any()
        assert not torch.isnan(field_map).any()
        
    except ImportError:
        pytest.skip("Data utilities not available")


if __name__ == "__main__":
    # Run basic tests
    processor = BezierParameterProcessor()
    sample_points = torch.tensor([[
        [[-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    ]], dtype=torch.float32)
    
    print("Testing BezierParameterProcessor...")
    
    # Basic functionality test
    density_map, field_map = processor(sample_points)
    print(f"✅ Basic forward pass: density {density_map.shape}, field {field_map.shape}")
    
    # Parameter count test
    param_count = processor.get_parameter_count()
    print(f"✅ Parameter count: {param_count / 1e6:.2f}M parameters")
    
    # Value range test
    assert density_map.min() >= 0.0 and density_map.max() <= 1.0
    assert field_map.min() >= -1.0 and field_map.max() <= 1.0
    print("✅ Output value ranges correct")
    
    print("All basic tests passed! ✅")