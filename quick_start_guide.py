#!/usr/bin/env python3
"""
EasyControl Quick Start Guide
Simple examples showing how to use EasyControl for different generation modes.
"""

import os
import sys
from pathlib import Path
from PIL import Image

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from enhanced_inference import EasyControlInference

def example_single_condition():
    """Example: Single condition generation."""
    print("=== Example 1: Single Condition Generation ===")
    
    # Initialize the inference pipeline
    inferencer = EasyControlInference(
        base_model_path="FLUX.1-dev",  # Path to your FLUX model
        device="cuda"  # Use "cpu" if no GPU available
    )
    
    # Generate with Canny edge control
    image = inferencer.single_condition_generate(
        prompt="A futuristic sports car driving through a neon-lit city at night",
        control_type="canny",  # Edge-based control
        control_image="test_imgs/canny.png",  # Your control image
        height=768,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=25,
        seed=42
    )
    
    # Save the result
    image.save("example_single_condition.png")
    print("✅ Single condition example saved to example_single_condition.png")

def example_multi_condition():
    """Example: Multi-condition generation."""
    print("\\n=== Example 2: Multi-Condition Generation ===")
    
    # Initialize the inference pipeline
    inferencer = EasyControlInference()
    
    # Generate with Subject + Spatial control
    image = inferencer.multi_condition_generate(
        prompt="A SKS person standing next to a vintage car on a mountain road",
        control_types=["subject", "canny"],  # Subject first, then spatial
        control_images=["test_imgs/subject_0.png", "test_imgs/canny.png"],
        height=768,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=25,
        seed=42,
        lora_weights=[1.0, 0.8]  # Different weights for each control
    )
    
    # Save the result
    image.save("example_multi_condition.png")
    print("✅ Multi-condition example saved to example_multi_condition.png")

def example_bezier_adapter():
    """Example: BezierAdapter generation for Chinese calligraphy."""
    print("\\n=== Example 3: BezierAdapter Generation ===")
    
    # Initialize the inference pipeline
    inferencer = EasyControlInference()
    
    # Find a sample bezier file
    bezier_dir = Path("bezier_curves_output_no_visualization/chinese-calligraphy-dataset")
    sample_files = list(bezier_dir.glob("吉/*_bezier.json"))  # Character '吉' (lucky)
    
    if not sample_files:
        print("⚠️  No bezier files found. Please run bezier_extraction.py first.")
        return
    
    sample_file = sample_files[0]
    print(f"Using bezier file: {sample_file}")
    
    # Generate with Bézier curve guidance
    image = inferencer.bezier_guided_generate(
        prompt="Traditional Chinese calligraphy character with elegant brushstrokes, artistic ink painting style",
        bezier_curves=str(sample_file),
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=25,
        seed=42
    )
    
    # Save the result
    image.save("example_bezier_adapter.png")
    print("✅ BezierAdapter example saved to example_bezier_adapter.png")

def example_style_transfer():
    """Example: Style transfer with Ghibli LoRA."""
    print("\\n=== Example 4: Style Transfer ===")
    
    # Initialize the inference pipeline
    inferencer = EasyControlInference()
    
    # Check if Ghibli model exists
    if not Path("models/Ghibli.safetensors").exists():
        print("⚠️  Ghibli.safetensors not found. Skipping style transfer example.")
        return
    
    # Generate with Ghibli style
    image = inferencer.single_condition_generate(
        prompt="Ghibli Studio style, Charming hand-drawn anime-style illustration of a peaceful countryside scene with rolling hills",
        control_type="ghibli",
        control_image="test_imgs/ghibli.png",  # Style reference image
        height=768,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=25,
        seed=42
    )
    
    # Save the result
    image.save("example_style_transfer.png")
    print("✅ Style transfer example saved to example_style_transfer.png")

def check_requirements():
    """Check if all requirements are available."""
    print("=== Checking Requirements ===")
    
    issues = []
    
    # Check base model
    if not Path("FLUX.1-dev").exists():
        issues.append("❌ FLUX.1-dev model not found")
    else:
        print("✅ FLUX.1-dev model found")
    
    # Check LoRA models
    models_dir = Path("models")
    if not models_dir.exists():
        issues.append("❌ Models directory not found")
    else:
        lora_files = list(models_dir.glob("*.safetensors"))
        print(f"✅ Found {len(lora_files)} LoRA models")
    
    # Check test images
    test_imgs_dir = Path("test_imgs")
    if not test_imgs_dir.exists():
        issues.append("❌ Test images directory not found")
    else:
        test_images = list(test_imgs_dir.glob("*.png"))
        print(f"✅ Found {len(test_images)} test images")
    
    # Check BezierAdapter data
    bezier_dir = Path("bezier_curves_output_no_visualization")
    if not bezier_dir.exists():
        issues.append("⚠️  BezierAdapter dataset not found (bezier examples will be skipped)")
    else:
        print("✅ BezierAdapter dataset found")
    
    if issues:
        print("\\nIssues found:")
        for issue in issues:
            print(f"  {issue}")
        print("\\nPlease resolve these issues before running examples.")
        return False
    
    print("\\n✅ All requirements satisfied!")
    return True

def main():
    """Run all examples."""
    print("=" * 60)
    print("EasyControl Quick Start Guide")
    print("=" * 60)
    
    # Check requirements
    if not check_requirements():
        print("\\n❌ Requirements not satisfied. Please set up the environment first.")
        print("\\nSetup steps:")
        print("1. Download FLUX.1-dev model")
        print("2. Download LoRA models from Hugging Face")
        print("3. Ensure test_imgs directory exists with sample images")
        print("4. Run bezier_extraction.py for BezierAdapter examples")
        return
    
    print("\\n" + "=" * 60)
    print("Running Examples")
    print("=" * 60)
    
    try:
        # Run examples
        if Path("test_imgs/canny.png").exists() and Path("models/canny.safetensors").exists():
            example_single_condition()
        else:
            print("⚠️  Skipping single condition example: Missing canny files")
        
        if (Path("test_imgs/subject_0.png").exists() and 
            Path("test_imgs/canny.png").exists() and
            Path("models/subject.safetensors").exists() and
            Path("models/canny.safetensors").exists()):
            example_multi_condition()
        else:
            print("⚠️  Skipping multi-condition example: Missing required files")
        
        if Path("bezier_curves_output_no_visualization").exists():
            example_bezier_adapter()
        else:
            print("⚠️  Skipping BezierAdapter example: Dataset not found")
        
        if Path("test_imgs/ghibli.png").exists() and Path("models/Ghibli.safetensors").exists():
            example_style_transfer()
        else:
            print("⚠️  Skipping style transfer example: Missing Ghibli files")
        
        print("\\n" + "=" * 60)
        print("Examples Complete!")
        print("=" * 60)
        print("\\nGenerated images:")
        for output_file in Path(".").glob("example_*.png"):
            print(f"  📸 {output_file}")
        
        print("\\n🎉 Quick start guide completed successfully!")
        print("\\nNext steps:")
        print("1. Explore the generated images")
        print("2. Modify prompts and parameters in the script")
        print("3. Try with your own control images")
        print("4. Train custom LoRA models for specific use cases")
        
    except Exception as e:
        print(f"\\n❌ Error during examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()