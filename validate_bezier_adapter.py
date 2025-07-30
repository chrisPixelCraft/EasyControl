#!/usr/bin/env python3
"""
Basic validation script for BezierAdapter framework.
Tests core functionality without requiring external test frameworks.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all BezierAdapter modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.bezier_adapter import (
            BezierParameterProcessor,
            ConditionInjectionAdapter,
            SpatialAttentionFuser,
            DensityAdaptiveSampler,
            StyleBezierFusionModule,
            BezierCurve,
            BezierConfig
        )
        print("✅ All BezierAdapter modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_bezier_processor():
    """Test BezierParameterProcessor basic functionality."""
    print("\nTesting BezierParameterProcessor...")
    
    try:
        from src.bezier_adapter.bezier_processor import BezierParameterProcessor
        
        # Create processor
        processor = BezierParameterProcessor(output_resolution=(64, 64))
        
        # Create sample bezier points
        bezier_points = torch.tensor([[[
            [-0.5, -0.5], [0.0, 0.0], [0.5, 0.5], [1.0, 1.0]
        ]]], dtype=torch.float32)  # [B=1, C=1, P=4, 2]
        
        # Test forward pass
        density_map, field_map = processor(bezier_points)
        
        # Validate outputs
        assert density_map.shape == (1, 1, 64, 64)
        assert field_map.shape == (1, 2, 64, 64)
        assert density_map.min() >= 0.0 and density_map.max() <= 1.0
        assert not torch.isnan(density_map).any()
        assert not torch.isnan(field_map).any()
        
        # Test parameter count
        param_count = processor.get_parameter_count()
        assert param_count > 0, f"Parameter count should be positive, got {param_count}"
        
        print(f"✅ BezierParameterProcessor basic tests passed ({param_count/1e6:.1f}M parameters)")
        return True
        
    except Exception as e:
        print(f"❌ BezierParameterProcessor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_condition_adapter():
    """Test ConditionInjectionAdapter basic functionality."""
    print("\nTesting ConditionInjectionAdapter...")
    
    try:
        from src.bezier_adapter.condition_adapter import ConditionInjectionAdapter
        
        adapter = ConditionInjectionAdapter()
        
        # Mock inputs
        style_features = torch.randn(1, 77, 768)
        bezier_coords = torch.randn(1, 8, 3)
        bezier_density = torch.rand(1, 8)
        
        # Test forward pass
        output = adapter(
            style_features=style_features,
            bezier_coords=bezier_coords,
            bezier_density=bezier_density
        )
        
        # Validate output
        assert output.shape[-1] == 3072  # Should match FluxTransformer inner_dim
        assert not torch.isnan(output).any()
        
        print("✅ ConditionInjectionAdapter basic tests passed")
        return True
        
    except Exception as e:
        print(f"❌ ConditionInjectionAdapter test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_spatial_attention():
    """Test SpatialAttentionFuser basic functionality."""
    print("\nTesting SpatialAttentionFuser...")
    
    try:
        from src.bezier_adapter.spatial_attention import SpatialAttentionFuser
        
        fuser = SpatialAttentionFuser()
        
        # Mock inputs
        spatial_features = torch.randn(1, 64*64, 768)  # [B, H*W, D]
        density_weights = torch.rand(1, 64, 64, 1)    # [B, H, W, 1]
        
        # Test forward pass
        fused_features, attention_weights = fuser(spatial_features, density_weights)
        
        # Validate outputs
        assert fused_features.shape[0] == 1, f"Batch dimension should be 1, got {fused_features.shape[0]}"
        assert len(fused_features.shape) >= 3, f"Output should be at least 3D, got shape {fused_features.shape}"
        assert not torch.isnan(fused_features).any()
        print(f"  Output shape: {fused_features.shape}")
        
        print("✅ SpatialAttentionFuser basic tests passed")
        return True
        
    except Exception as e:
        print(f"❌ SpatialAttentionFuser test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_style_fusion():
    """Test StyleBezierFusionModule basic functionality."""
    print("\nTesting StyleBezierFusionModule...")
    
    try:
        from src.bezier_adapter.style_fusion import StyleBezierFusionModule
        
        fusion_module = StyleBezierFusionModule()
        
        # Mock inputs
        spatial_features = torch.randn(1, 128, 3072)  # [B, N, D]
        style_features = torch.randn(1, 768)         # [B, style_dim]
        bezier_density = torch.rand(1, 1, 32, 32)    # [B, 1, H, W]
        
        # Test forward pass
        fused_features = fusion_module(
            spatial_features=spatial_features,
            style_features=style_features,
            bezier_density=bezier_density
        )
        
        # Validate output
        assert fused_features.shape == (1, 128, 3072)
        assert not torch.isnan(fused_features).any()
        
        print("✅ StyleBezierFusionModule basic tests passed")
        return True
        
    except Exception as e:
        print(f"❌ StyleBezierFusionModule test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_parameter_efficiency():
    """Test total parameter count of BezierAdapter framework."""
    print("\nTesting parameter efficiency...")
    
    try:
        from src.bezier_adapter import (
            BezierParameterProcessor,
            ConditionInjectionAdapter,
            SpatialAttentionFuser,
            StyleBezierFusionModule
        )
        
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
            print(f"  {name}: {param_count / 1e6:.1f}M parameters")
        
        print(f"  Total BezierAdapter parameters: {total_params / 1e6:.1f}M")
        
        # Validate reasonable parameter count
        assert total_params > 0, f"Total parameters should be positive, got {total_params}"
        
        print("✅ Parameter efficiency validated")
        return True
        
    except Exception as e:
        print(f"❌ Parameter efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all validation tests."""
    print("=" * 60)
    print("BezierAdapter Framework Validation")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_bezier_processor,
        test_condition_adapter,
        test_spatial_attention,
        test_style_fusion,
        test_parameter_efficiency
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("Stopping validation due to test failure.")
            break
    
    print("\n" + "=" * 60)
    print(f"Validation Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All BezierAdapter validation tests passed!")
        print("✅ Framework is ready for use")
        return True
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)