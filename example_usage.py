#!/usr/bin/env python3
"""
Example Usage: Complete pipeline from calligraphy images to FLUX-compatible density maps

This example demonstrates how to:
1. Extract Bézier curves from calligraphy images using BezierCurveExtractor
2. Process them into density maps using BezierParameterProcessor
3. Use the density maps for FLUX transformer conditioning
4. Generate calligraphy-guided images using the BezierFluxPipeline

Prerequisites:
- Install required dependencies: torch, torchvision, opencv-python, numpy, matplotlib
- Have the EasyControl FLUX model weights
- Place your calligraphy images in the input directory
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.append('src')

# Import our modules
from bezier_extraction import BezierCurveExtractor
from bezier_parameter_processor import BezierParameterProcessor, create_bezier_processor
from bezier_pipeline_integration import BezierFluxPipeline, create_bezier_flux_pipeline
from pipeline import FluxPipeline

def example_complete_pipeline():
    """
    Complete example showing the entire pipeline from image to generation.
    """
    print("=" * 70)
    print("COMPLETE BEZIER-GUIDED CALLIGRAPHY GENERATION PIPELINE")
    print("=" * 70)

    # Step 1: Create or load test calligraphy image
    print("\n1. PREPARING CALLIGRAPHY IMAGE")
    print("-" * 40)

    # Create a sample calligraphy image
    calligraphy_img = create_sample_calligraphy()
    input_path = "sample_calligraphy.png"
    cv2.imwrite(input_path, calligraphy_img)
    print(f"✓ Created test calligraphy image: {input_path}")

    # Step 2: Extract Bézier curves
    print("\n2. EXTRACTING BÉZIER CURVES")
    print("-" * 40)

    extractor = BezierCurveExtractor(
        smoothing_factor=0.001,
        max_points=300,
        curve_resolution=150,
        max_segments=30
    )

    bezier_data = extractor.extract_character_bezier(input_path)
    print(f"✓ Found {len(bezier_data['characters'])} characters")

    # Save and visualize Bézier curves
    bezier_json_path = "sample_bezier_curves.json"
    extractor.save_bezier_data(bezier_data, bezier_json_path)

    bezier_viz_path = "bezier_visualization.png"
    extractor.visualize_bezier_curves(input_path, bezier_data, bezier_viz_path)
    print(f"✓ Saved Bézier data: {bezier_json_path}")
    print(f"✓ Created visualization: {bezier_viz_path}")

    # Step 3: Process to density maps
    print("\n3. GENERATING DENSITY MAPS")
    print("-" * 40)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create BezierParameterProcessor
    bezier_processor = create_bezier_processor(device=device)

    # Count parameters
    total_params = sum(p.numel() for p in bezier_processor.parameters())
    print(f"BezierParameterProcessor parameters: {total_params:,}")

    # Generate density maps
    density_maps = bezier_processor(bezier_data)
    print(f"✓ Generated density maps: {density_maps.shape}")
    print(f"  Range: [{density_maps.min():.3f}, {density_maps.max():.3f}]")

    # Visualize density maps
    density_viz_path = "density_visualization.png"
    density_colored = bezier_processor.visualize_density_map(density_maps[0], density_viz_path)
    print(f"✓ Created density visualization: {density_viz_path}")

    # Step 4: Create comprehensive visualization
    print("\n4. CREATING COMPREHENSIVE VISUALIZATION")
    print("-" * 40)

    create_pipeline_visualization(
        calligraphy_img,
        extractor,
        bezier_data,
        bezier_processor,
        density_maps,
        input_path
    )

    print("✓ Created comprehensive visualization: pipeline_visualization.png")

    # Step 5: (Optional) FLUX Generation Example
    print("\n5. FLUX GENERATION EXAMPLE")
    print("-" * 40)
    print("Note: This requires the EasyControl FLUX model weights")
    print("Uncomment the following section to test with actual FLUX model:")

    # Uncomment for actual FLUX generation:
    # flux_generation_example(bezier_data, bezier_processor)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("Generated files:")
    print(f"  - {input_path} (original calligraphy)")
    print(f"  - {bezier_json_path} (extracted Bézier curves)")
    print(f"  - {bezier_viz_path} (Bézier visualization)")
    print(f"  - {density_viz_path} (density map visualization)")
    print(f"  - pipeline_visualization.png (complete pipeline)")

def create_sample_calligraphy():
    """Create a sample calligraphy image for testing."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Draw Chinese-style calligraphy characters
    # Character 1: 龙 (dragon)
    cv2.ellipse(img, (150, 150), (40, 80), 0, 0, 360, (0, 0, 0), 4)
    cv2.line(img, (110, 120), (190, 120), (0, 0, 0), 3)
    cv2.line(img, (130, 100), (170, 200), (0, 0, 0), 3)
    cv2.line(img, (120, 180), (180, 140), (0, 0, 0), 2)

    # Character 2: 书 (book)
    cv2.line(img, (280, 100), (380, 100), (0, 0, 0), 3)
    cv2.line(img, (300, 120), (360, 120), (0, 0, 0), 2)
    cv2.line(img, (320, 80), (320, 200), (0, 0, 0), 4)
    cv2.line(img, (290, 160), (370, 160), (0, 0, 0), 2)
    cv2.line(img, (290, 190), (370, 190), (0, 0, 0), 2)

    # Character 3: 法 (method)
    cv2.line(img, (450, 90), (550, 90), (0, 0, 0), 3)
    cv2.ellipse(img, (480, 140), (25, 35), 0, 0, 360, (0, 0, 0), 2)
    cv2.line(img, (520, 120), (520, 200), (0, 0, 0), 3)
    cv2.line(img, (470, 170), (550, 170), (0, 0, 0), 2)

    # Add some artistic flourishes
    cv2.ellipse(img, (100, 300), (50, 30), 45, 0, 180, (0, 0, 0), 2)
    cv2.ellipse(img, (500, 300), (40, 25), -30, 0, 180, (0, 0, 0), 2)

    return img

def create_pipeline_visualization(calligraphy_img, extractor, bezier_data,
                                bezier_processor, density_maps, input_path):
    """Create a comprehensive visualization of the entire pipeline."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original calligraphy image
    axes[0, 0].imshow(cv2.cvtColor(calligraphy_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. Original Calligraphy", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Bézier curve extraction
    bezier_viz = extractor.visualize_bezier_curves(input_path, bezier_data)
    axes[0, 1].imshow(cv2.cvtColor(bezier_viz, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("2. Extracted Bézier Curves", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    # Density map (raw)
    density_raw = density_maps[0, 0].cpu().detach().numpy()
    im1 = axes[0, 2].imshow(density_raw, cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title("3. Generated Density Map", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Density map (colored)
    density_colored = bezier_processor.visualize_density_map(density_maps[0])
    axes[1, 0].imshow(cv2.cvtColor(density_colored, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("4. Colored Density Map", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')

    # Statistics visualization
    axes[1, 1].bar(['Characters', 'Total Curves', 'Avg Points/Curve'],
                   [len(bezier_data['characters']),
                    sum(len(char['bezier_curves']) for char in bezier_data['characters']),
                    np.mean([len(curve) for char in bezier_data['characters']
                            for curve in char['bezier_curves']])],
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title("5. Extraction Statistics", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Count')

    # Density distribution
    density_flat = density_raw.flatten()
    axes[1, 2].hist(density_flat, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 2].set_title("6. Density Distribution", fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Density Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pipeline_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

def flux_generation_example(bezier_data, bezier_processor):
    """
    Example of using the BezierFluxPipeline for actual generation.
    Note: This requires the EasyControl FLUX model weights.
    """
    print("Loading FLUX model...")

    # Load the original FLUX pipeline
    # Note: Replace with actual model loading code
    # flux_pipeline = FluxPipeline.from_pretrained("path/to/flux/model")

    # Create Bézier-enabled pipeline
    # bezier_pipeline = create_bezier_flux_pipeline(
    #     flux_pipeline=flux_pipeline,
    #     bezier_processor=bezier_processor,
    #     enable_bezier_conditioning=True
    # )

    # Generate images with Bézier conditioning
    # prompt = "A beautiful Chinese calligraphy artwork, traditional ink painting style"
    # images = bezier_pipeline(
    #     prompt=prompt,
    #     bezier_data=bezier_data,
    #     bezier_scale=1.0,
    #     height=512,
    #     width=512,
    #     num_inference_steps=20,
    #     guidance_scale=7.5
    # )

    # Save generated images
    # for i, image in enumerate(images["images"]):
    #     image.save(f"generated_calligraphy_{i}.png")

    print("Note: Actual FLUX generation requires model weights")
    print("Replace the commented code above with actual model loading")

def batch_processing_example():
    """
    Example of batch processing multiple calligraphy images.
    """
    print("\n" + "=" * 70)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 70)

    # Create multiple test images
    test_images = []
    bezier_data_list = []

    extractor = BezierCurveExtractor()

    for i in range(3):
        # Create different calligraphy samples
        img = create_sample_calligraphy()
        # Add some variation
        img = cv2.addWeighted(img, 0.8, np.random.randint(0, 50, img.shape, dtype=np.uint8), 0.2, 0)

        img_path = f"test_batch_{i}.png"
        cv2.imwrite(img_path, img)

        # Extract Bézier curves
        bezier_data = extractor.extract_character_bezier(img_path)
        bezier_data_list.append(bezier_data)

        print(f"✓ Processed image {i+1}: {len(bezier_data['characters'])} characters")

    # Batch process with BezierParameterProcessor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bezier_processor = create_bezier_processor(device=device)

    # Process all at once
    batch_density_maps = bezier_processor(bezier_data_list)
    print(f"✓ Generated batch density maps: {batch_density_maps.shape}")

    # Visualize batch results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in range(3):
        density_map = batch_density_maps[i, 0].cpu().detach().numpy()
        axes[i].imshow(density_map, cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(f"Batch Sample {i+1}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig("batch_density_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Created batch visualization: batch_density_visualization.png")

def performance_benchmark():
    """
    Performance benchmark for the BezierParameterProcessor.
    """
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    bezier_processor = create_bezier_processor(device=device)

    # Test different complexities
    test_cases = [
        {"chars": 1, "curves": 2, "points": 4},
        {"chars": 5, "curves": 10, "points": 4},
        {"chars": 10, "curves": 20, "points": 6},
        {"chars": 20, "curves": 40, "points": 8}
    ]

    print(f"Device: {device}")
    print("Testing different complexities...")

    import time

    for i, case in enumerate(test_cases):
        # Create mock data
        bezier_data = {
            'image_path': 'test.png',
            'characters': []
        }

        for j in range(case['chars']):
            char_data = {
                'character_id': j,
                'contour_area': 100.0,
                'bounding_box': (50, 50, 100, 100),
                'bezier_curves': [],
                'original_contour_points': 50
            }

            for k in range(case['curves']):
                control_points = np.random.rand(case['points'], 2) * 100
                char_data['bezier_curves'].append(control_points.tolist())

            bezier_data['characters'].append(char_data)

        # Benchmark
        start_time = time.time()

        # Warmup
        for _ in range(3):
            _ = bezier_processor(bezier_data)

        # Actual benchmark
        times = []
        for _ in range(10):
            start = time.time()
            density_map = bezier_processor(bezier_data)
            end = time.time()
            times.append(end - start)

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"Case {i+1}: {case['chars']} chars, {case['curves']} curves/char")
        print(f"  Average time: {avg_time:.4f} ± {std_time:.4f} seconds")
        print(f"  Output shape: {density_map.shape}")
        print(f"  Memory usage: {density_map.numel() * 4 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    print("BEZIER PARAMETER PROCESSOR - COMPLETE USAGE EXAMPLE")
    print("=" * 70)

    try:
        # Main pipeline example
        example_complete_pipeline()

        # Batch processing example
        batch_processing_example()

        # Performance benchmark
        performance_benchmark()

        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)