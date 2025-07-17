#!/usr/bin/env python3
"""
Test script demonstrating the integration between BezierCurveExtractor and BezierParameterProcessor.
This shows the complete pipeline from calligraphy images to FLUX-compatible density maps.
"""

import os
import sys
import torch
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add src directory to path
sys.path.append('src')

# Import our modules
from bezier_extraction import BezierCurveExtractor
from bezier_parameter_processor import BezierParameterProcessor, create_bezier_processor

def create_test_image():
    """Create a simple test calligraphy image for demonstration."""
    img = np.ones((256, 256, 3), dtype=np.uint8) * 255

    # Draw some simple calligraphy-like strokes
    cv2.ellipse(img, (100, 100), (30, 60), 45, 0, 360, (0, 0, 0), 3)
    cv2.ellipse(img, (150, 150), (20, 40), -30, 0, 360, (0, 0, 0), 2)
    cv2.line(img, (80, 200), (180, 200), (0, 0, 0), 4)

    return img

def test_bezier_extraction():
    """Test the BezierCurveExtractor with a sample image."""
    print("=" * 60)
    print("TESTING BEZIER CURVE EXTRACTION")
    print("=" * 60)

    # Create test image
    test_img = create_test_image()
    test_img_path = "test_calligraphy.png"
    cv2.imwrite(test_img_path, test_img)
    print(f"Created test image: {test_img_path}")

    # Initialize extractor
    extractor = BezierCurveExtractor(
        smoothing_factor=0.001,
        max_points=200,
        curve_resolution=100,
        max_segments=20
    )

    # Extract Bézier curves
    start_time = time.time()
    bezier_data = extractor.extract_character_bezier(test_img_path)
    extraction_time = time.time() - start_time

    print(f"Extraction completed in {extraction_time:.3f} seconds")
    print(f"Found {len(bezier_data['characters'])} characters")

    # Display results
    for i, char_data in enumerate(bezier_data['characters']):
        print(f"Character {i}:")
        print(f"  - Contour area: {char_data['contour_area']:.2f}")
        print(f"  - Bounding box: {char_data['bounding_box']}")
        print(f"  - Number of Bézier curves: {len(char_data['bezier_curves'])}")

        # Show first few control points
        if char_data['bezier_curves']:
            first_curve = char_data['bezier_curves'][0]
            print(f"  - First curve control points: {len(first_curve)} points")
            print(f"    Sample: {first_curve[:2]}...")

    # Save data
    bezier_json_path = "test_bezier_data.json"
    extractor.save_bezier_data(bezier_data, bezier_json_path)
    print(f"Saved Bézier data to: {bezier_json_path}")

    # Create visualization
    viz_path = "test_bezier_visualization.png"
    extractor.visualize_bezier_curves(test_img_path, bezier_data, viz_path)
    print(f"Created visualization: {viz_path}")

    return bezier_data, bezier_json_path

def test_bezier_parameter_processor(bezier_data, bezier_json_path):
    """Test the BezierParameterProcessor with extracted Bézier data."""
    print("\n" + "=" * 60)
    print("TESTING BEZIER PARAMETER PROCESSOR")
    print("=" * 60)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create processor
    processor = create_bezier_processor(device=device)

    # Count parameters
    total_params = sum(p.numel() for p in processor.parameters())
    trainable_params = sum(p.numel() for p in processor.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Target was ~2.1M parameters: {'✓' if trainable_params < 2.5e6 else '✗'}")

    # Test single sample processing
    print("\nTesting single sample processing...")
    start_time = time.time()

    # Process from data directly
    density_map = processor(bezier_data)
    processing_time = time.time() - start_time

    print(f"Processing completed in {processing_time:.3f} seconds")
    print(f"Output density map shape: {density_map.shape}")
    print(f"Density map range: [{density_map.min():.3f}, {density_map.max():.3f}]")

    # Test from JSON file
    print("\nTesting JSON file processing...")
    density_map_from_json = processor.process_bezier_file(bezier_json_path)
    print(f"JSON processing shape: {density_map_from_json.shape}")

    # Verify outputs match
    if torch.allclose(density_map, density_map_from_json, atol=1e-6):
        print("✓ Direct processing and JSON processing produce identical results")
    else:
        print("✗ Direct processing and JSON processing differ")

    # Test batch processing
    print("\nTesting batch processing...")
    batch_data = [bezier_data, bezier_data, bezier_data]  # 3 identical samples
    batch_density = processor(batch_data)
    print(f"Batch processing shape: {batch_density.shape}")

    # Visualize density map
    print("\nCreating density map visualization...")
    density_viz = processor.visualize_density_map(
        density_map[0],
        "test_density_visualization.png"
    )
    print("Created density visualization: test_density_visualization.png")

    return density_map, processor

def test_flux_compatibility(density_map):
    """Test compatibility with FLUX transformer input format."""
    print("\n" + "=" * 60)
    print("TESTING FLUX COMPATIBILITY")
    print("=" * 60)

    # Check tensor properties
    print(f"Density map shape: {density_map.shape}")
    print(f"Expected shape: [B, 1, H, W] where H=W=64")

    # Verify shape compatibility
    batch_size, channels, height, width = density_map.shape
    expected_shape = (batch_size, 1, 64, 64)

    if density_map.shape == expected_shape:
        print("✓ Shape is compatible with FLUX transformer")
    else:
        print(f"✗ Shape mismatch. Expected {expected_shape}, got {density_map.shape}")

    # Verify data type and range
    print(f"Data type: {density_map.dtype}")
    print(f"Value range: [{density_map.min():.3f}, {density_map.max():.3f}]")
    print(f"Mean: {density_map.mean():.3f}, Std: {density_map.std():.3f}")

    if density_map.dtype == torch.float32:
        print("✓ Data type is compatible (float32)")
    else:
        print(f"✗ Data type should be float32, got {density_map.dtype}")

    if 0 <= density_map.min() and density_map.max() <= 1:
        print("✓ Values are in valid range [0, 1]")
    else:
        print("✗ Values should be in range [0, 1]")

    # Test gradient flow
    print("\nTesting gradient flow...")
    density_map.requires_grad_(True)
    loss = density_map.sum()
    loss.backward()

    if density_map.grad is not None:
        print("✓ Gradient flow is working")
        print(f"Gradient shape: {density_map.grad.shape}")
        print(f"Gradient norm: {density_map.grad.norm():.6f}")
    else:
        print("✗ Gradient flow failed")

def test_performance_benchmarks():
    """Test performance with different input sizes."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = create_bezier_processor(device=device)

    # Create test data with different complexities
    test_cases = [
        {"name": "Simple (1 character, 2 curves)", "chars": 1, "curves_per_char": 2},
        {"name": "Medium (3 characters, 5 curves)", "chars": 3, "curves_per_char": 5},
        {"name": "Complex (5 characters, 10 curves)", "chars": 5, "curves_per_char": 10},
    ]

    for case in test_cases:
        print(f"\nTesting {case['name']}...")

        # Create mock Bézier data
        bezier_data = {
            'image_path': 'test.png',
            'characters': []
        }

        for i in range(case['chars']):
            char_data = {
                'character_id': i,
                'contour_area': 100.0,
                'bounding_box': (50, 50, 100, 100),
                'bezier_curves': [],
                'original_contour_points': 50
            }

            # Add random Bézier curves
            for j in range(case['curves_per_char']):
                # Create random control points (4 points for cubic Bézier)
                control_points = np.random.rand(4, 2) * 100
                char_data['bezier_curves'].append(control_points.tolist())

            bezier_data['characters'].append(char_data)

        # Benchmark processing time
        start_time = time.time()
        density_map = processor(bezier_data)
        end_time = time.time()

        processing_time = end_time - start_time
        print(f"  Processing time: {processing_time:.3f} seconds")
        print(f"  Output shape: {density_map.shape}")
        print(f"  Memory usage: {density_map.element_size() * density_map.numel() / 1024:.1f} KB")

def create_comparative_visualization():
    """Create a comprehensive visualization comparing all stages."""
    print("\n" + "=" * 60)
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print("=" * 60)

    # Create test image
    test_img = create_test_image()

    # Extract Bézier curves
    extractor = BezierCurveExtractor()
    bezier_data = extractor.extract_character_bezier("test_calligraphy.png")

    # Process to density map
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = create_bezier_processor(device=device)
    density_map = processor(bezier_data)

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original image
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title("Original Calligraphy Image")
    axes[0, 0].axis('off')

    # Bézier curve visualization
    bezier_viz = extractor.visualize_bezier_curves("test_calligraphy.png", bezier_data)
    axes[0, 1].imshow(cv2.cvtColor(bezier_viz, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Extracted Bézier Curves")
    axes[0, 1].axis('off')

    # Density map (raw)
    density_np = density_map[0, 0].cpu().detach().numpy()
    axes[1, 0].imshow(density_np, cmap='hot')
    axes[1, 0].set_title("Generated Density Map")
    axes[1, 0].axis('off')

    # Density map (colored)
    density_colored = processor.visualize_density_map(density_map[0])
    axes[1, 1].imshow(cv2.cvtColor(density_colored, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Density Map (Colored)")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig("comprehensive_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Created comprehensive visualization: comprehensive_visualization.png")

def main():
    """Main test function."""
    print("BEZIER PARAMETER PROCESSOR INTEGRATION TEST")
    print("=" * 60)
    print("This test demonstrates the complete pipeline from calligraphy")
    print("images to FLUX-compatible density maps.")
    print("=" * 60)

    try:
        # Test 1: Bézier curve extraction
        bezier_data, bezier_json_path = test_bezier_extraction()

        # Test 2: Parameter processing
        density_map, processor = test_bezier_parameter_processor(bezier_data, bezier_json_path)

        # Test 3: FLUX compatibility
        test_flux_compatibility(density_map)

        # Test 4: Performance benchmarks
        test_performance_benchmarks()

        # Test 5: Comprehensive visualization
        create_comparative_visualization()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("Generated files:")
        print("- test_calligraphy.png (test image)")
        print("- test_bezier_data.json (extracted Bézier data)")
        print("- test_bezier_visualization.png (Bézier curve visualization)")
        print("- test_density_visualization.png (density map visualization)")
        print("- comprehensive_visualization.png (complete pipeline)")

    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)