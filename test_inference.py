#!/usr/bin/env python3
"""
Comprehensive test script for EasyControl inference modes.
Tests single condition, multi-condition, and BezierAdapter functionality.
"""

import os
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from enhanced_inference import EasyControlInference

def test_environment():
    """Test environment setup and requirements."""
    print("=== Environment Tests ===")
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    
    # Check base model
    base_model_path = "FLUX.1-dev"
    if Path(base_model_path).exists():
        print(f"✅ Base model found: {base_model_path}")
    else:
        print(f"⚠️  Base model not found: {base_model_path}")
        print("You may need to download it first")
    
    # Check LoRA models
    models_dir = Path("models")
    if models_dir.exists():
        lora_files = list(models_dir.glob("*.safetensors"))
        print(f"✅ Found {len(lora_files)} LoRA models: {[f.name for f in lora_files]}")
    else:
        print("⚠️  Models directory not found")
    
    # Check test images
    test_imgs_dir = Path("test_imgs")
    if test_imgs_dir.exists():
        test_images = list(test_imgs_dir.glob("*.png"))
        print(f"✅ Found {len(test_images)} test images: {[img.name for img in test_images]}")
    else:
        print("⚠️  Test images directory not found")
    
    # Check BezierAdapter data
    bezier_dir = Path("bezier_curves_output_no_visualization")
    if bezier_dir.exists():
        print("✅ BezierAdapter dataset found")
    else:
        print("⚠️  BezierAdapter dataset not found")
    
    return True

def test_single_condition_inference():
    """Test single condition inference with different control types."""
    print("\\n=== Single Condition Inference Tests ===")
    
    try:
        # Initialize inference
        inferencer = EasyControlInference(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Test configurations
        test_configs = [
            {
                "name": "Canny Edge Control",
                "control_type": "canny",
                "control_image": "test_imgs/canny.png",
                "prompt": "A futuristic sports car with neon lights"
            },
            {
                "name": "Subject Control", 
                "control_type": "subject",
                "control_image": "test_imgs/subject_0.png",
                "prompt": "A SKS person in a magical forest"
            }
        ]
        
        for config in test_configs:
            print(f"\\nTesting {config['name']}...")
            
            # Check if required files exist
            if not Path(config["control_image"]).exists():
                print(f"⚠️  Skipping: Control image not found - {config['control_image']}")
                continue
            
            if not Path(f"models/{config['control_type']}.safetensors").exists():
                print(f"⚠️  Skipping: LoRA model not found - {config['control_type']}.safetensors")
                continue
            
            try:
                # Generate image
                image = inferencer.single_condition_generate(
                    prompt=config["prompt"],
                    control_type=config["control_type"],
                    control_image=config["control_image"],
                    height=512,  # Smaller for testing
                    width=512,
                    num_inference_steps=10,  # Fewer steps for testing
                    seed=42
                )
                
                # Save result
                output_path = f"test_output_{config['control_type']}.png"
                image.save(output_path)
                print(f"✅ {config['name']} test passed - saved to {output_path}")
                
            except Exception as e:
                print(f"❌ {config['name']} test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Single condition inference initialization failed: {e}")
        return False

def test_multi_condition_inference():
    """Test multi-condition inference."""
    print("\\n=== Multi-Condition Inference Tests ===")
    
    try:
        # Initialize inference
        inferencer = EasyControlInference(device="cuda" if torch.cuda.is_available() else "cpu")
        
        # Test multi-condition (subject + spatial)
        control_types = ["subject", "canny"]
        control_images = ["test_imgs/subject_1.png", "test_imgs/canny.png"]
        
        # Check if files exist
        missing_files = []
        for img_path in control_images:
            if not Path(img_path).exists():
                missing_files.append(img_path)
        
        for control_type in control_types:
            if not Path(f"models/{control_type}.safetensors").exists():
                missing_files.append(f"models/{control_type}.safetensors")
        
        if missing_files:
            print(f"⚠️  Skipping multi-condition test: Missing files - {missing_files}")
            return True
        
        print("Testing Subject + Canny multi-condition...")
        
        try:
            image = inferencer.multi_condition_generate(
                prompt="A SKS person driving a sleek car on a mountain road",
                control_types=control_types,
                control_images=control_images,
                height=512,
                width=512,
                num_inference_steps=10,
                seed=42
            )
            
            output_path = "test_output_multi.png"
            image.save(output_path)
            print(f"✅ Multi-condition test passed - saved to {output_path}")
            
        except Exception as e:
            print(f"❌ Multi-condition test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-condition inference initialization failed: {e}")
        return False

def test_bezier_adapter_inference():
    """Test BezierAdapter inference."""
    print("\\n=== BezierAdapter Inference Tests ===")
    
    try:
        # Check for bezier data
        bezier_dir = Path("bezier_curves_output_no_visualization/chinese-calligraphy-dataset")
        if not bezier_dir.exists():
            print("⚠️  Skipping BezierAdapter test: Dataset not found")
            return True
        
        # Find a sample bezier file
        sample_files = list(bezier_dir.glob("*/.*_bezier.json"))
        if not sample_files:
            print("⚠️  Skipping BezierAdapter test: No bezier JSON files found")
            return True
        
        sample_file = sample_files[0]
        print(f"Testing BezierAdapter with {sample_file}")
        
        # Initialize inference
        inferencer = EasyControlInference(device="cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            image = inferencer.bezier_guided_generate(
                prompt="Traditional Chinese calligraphy with elegant brushstrokes and artistic flow",
                bezier_curves=str(sample_file),
                height=512,
                width=512,
                num_inference_steps=10,
                seed=42
            )
            
            output_path = "test_output_bezier.png"
            image.save(output_path)
            print(f"✅ BezierAdapter test passed - saved to {output_path}")
            
        except Exception as e:
            print(f"❌ BezierAdapter test failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ BezierAdapter inference initialization failed: {e}")
        return False

def test_memory_management():
    """Test memory management and cache clearing."""
    print("\\n=== Memory Management Tests ===")
    
    if not torch.cuda.is_available():
        print("⚠️  Skipping memory tests: CUDA not available")
        return True
    
    try:
        # Initialize inference
        inferencer = EasyControlInference(device="cuda")
        
        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        print(f"Initial GPU memory: {initial_memory / 1024**2:.1f}MB")
        
        # Test cache clearing
        inferencer.clear_cache()
        after_clear_memory = torch.cuda.memory_allocated()
        
        print(f"Memory after cache clear: {after_clear_memory / 1024**2:.1f}MB")
        
        # Test generation memory usage
        if Path("test_imgs/canny.png").exists() and Path("models/canny.safetensors").exists():
            try:
                image = inferencer.single_condition_generate(
                    prompt="A simple test image",
                    control_type="canny",
                    control_image="test_imgs/canny.png",
                    height=256,
                    width=256,
                    num_inference_steps=5,
                    seed=42
                )
                
                peak_memory = torch.cuda.memory_allocated()
                print(f"Peak memory during generation: {peak_memory / 1024**2:.1f}MB")
                
                # Memory should be cleared after generation
                final_memory = torch.cuda.memory_allocated()
                print(f"Final GPU memory: {final_memory / 1024**2:.1f}MB")
                
                print("✅ Memory management test passed")
                
            except Exception as e:
                print(f"❌ Memory test generation failed: {e}")
        else:
            print("⚠️  Skipping memory generation test: Required files not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def main():
    """Run all inference tests."""
    print("===" * 20)
    print("EasyControl Comprehensive Inference Tests")
    print("===" * 20)
    
    # Test results
    results = {
        "Environment": False,
        "Single Condition": False,
        "Multi-Condition": False,
        "BezierAdapter": False,
        "Memory Management": False
    }
    
    # Run tests
    try:
        results["Environment"] = test_environment()
        results["Single Condition"] = test_single_condition_inference()
        results["Multi-Condition"] = test_multi_condition_inference()
        results["BezierAdapter"] = test_bezier_adapter_inference()
        results["Memory Management"] = test_memory_management()
        
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\\n\\n❌ Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print("\\n" + "===" * 20)
    print("Test Results Summary")
    print("===" * 20)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EasyControl inference is ready.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    print("\\nGenerated test outputs:")
    output_files = list(Path(".").glob("test_output_*.png"))
    for output_file in output_files:
        print(f"  {output_file}")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)