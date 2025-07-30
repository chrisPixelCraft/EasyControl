#!/usr/bin/env python3
"""
Fix BezierAdapter Training Issues
Validates bezier data and creates memory-optimized training configuration.
"""

import json
import os
import sys
from pathlib import Path
import torch

def validate_bezier_files(dataset_path):
    """Validate all bezier JSON files and filter out corrupted ones."""
    print("=== Validating Bezier Dataset ===")
    
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_path}")
        return []
    
    valid_files = []
    invalid_files = []
    
    # Find all bezier JSON files
    bezier_files = list(dataset_dir.glob("**/*_bezier.json"))
    print(f"Found {len(bezier_files)} bezier JSON files")
    
    for file_path in bezier_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate structure
            required_fields = ['characters']
            if all(field in data for field in required_fields):
                # Check if has actual bezier curve data
                if data['characters'] and len(data['characters']) > 0:
                    has_curves = False
                    for char in data['characters']:
                        if 'bezier_curves' in char and char['bezier_curves']:
                            has_curves = True
                            break
                    
                    if has_curves:
                        valid_files.append(file_path)
                    else:
                        invalid_files.append((file_path, "No bezier curves found"))
                else:
                    invalid_files.append((file_path, "No character data"))
            else:
                invalid_files.append((file_path, "Missing required fields"))
                
        except json.JSONDecodeError as e:
            invalid_files.append((file_path, f"JSON decode error: {e}"))
        except Exception as e:
            invalid_files.append((file_path, f"Error: {e}"))
    
    print(f"✅ Valid files: {len(valid_files)}")
    print(f"❌ Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print("\\nInvalid files:")
        for file_path, reason in invalid_files[:10]:  # Show first 10
            print(f"  {file_path.name}: {reason}")
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files) - 10} more")
    
    return valid_files

def create_clean_training_data(valid_files, output_path, max_samples=50):
    """Create clean training data from valid bezier files, mapping to original JPG images."""
    print(f"\\n=== Creating Clean Training Data (max {max_samples} samples) ===")
    
    training_data = []
    skipped_files = []
    
    for file_path in valid_files[:max_samples]:
        try:
            # Extract the base filename without '_bezier.json'
            base_filename = file_path.stem.replace('_bezier', '')
            character = file_path.parent.name
            
            # Map to the original JPG image path
            original_image_path = Path(f"chinese-calligraphy-dataset/chinese-calligraphy-dataset/{character}/{base_filename}.jpg")
            
            # Check if the original image exists
            if original_image_path.exists():
                # Double-check file is readable
                try:
                    from PIL import Image
                    with Image.open(original_image_path) as img:
                        img.verify()  # Verify it's a valid image
                except Exception as verify_error:
                    skipped_files.append((file_path, f"Image verification failed: {verify_error}"))
                    continue
                training_entry = {
                    "bezier_curves": str(original_image_path),  # Point to JPG image, not JSON
                    "caption": f"Traditional Chinese calligraphy character '{character}' with elegant brushstrokes and precise form",
                    "target": str(original_image_path),  # Same image for target
                    "character": character,
                    "bezier_data": str(file_path.relative_to(Path(".")))  # Keep reference to JSON for future use
                }
                training_data.append(training_entry)
            else:
                skipped_files.append((file_path, f"Original image not found: {original_image_path}"))
            
        except Exception as e:
            skipped_files.append((file_path, f"Error processing: {e}"))
    
    # Report skipped files
    if skipped_files:
        print(f"⚠️  Skipped {len(skipped_files)} files:")
        for file_path, reason in skipped_files[:5]:  # Show first 5
            print(f"   {file_path.name}: {reason}")
        if len(skipped_files) > 5:
            print(f"   ... and {len(skipped_files) - 5} more")
    
    # Write clean JSONL file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✅ Created clean training data: {output_path}")
    print(f"   Total samples: {len(training_data)}")
    print(f"   Successfully mapped {len(training_data)} bezier files to JPG images")
    return len(training_data)

def get_gpu_memory_info():
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return "No CUDA available"
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    free = total_memory - allocated
    
    return {
        "total": total_memory,
        "allocated": allocated,
        "reserved": reserved,
        "free": free
    }

def create_memory_optimized_config(gpu_memory_gb):
    """Create memory-optimized training configuration based on available GPU memory."""
    print(f"\\n=== Creating Memory-Optimized Configuration ===")
    print(f"Available GPU memory: {gpu_memory_gb:.1f}GB")
    
    # Memory-optimized parameters based on available memory
    # Even more aggressive optimization for FLUX.1-dev which is memory-intensive
    if gpu_memory_gb < 50:
        # For 40GB GPU - use very conservative settings
        config = {
            "cond_size": 256,
            "noise_size": 512,
            "num_epochs": 20,
            "validation_steps": 20,  # Less frequent validation
            "checkpointing_steps": 20,
            "test_h": 512,
            "test_w": 512,
            "gradient_checkpointing": False,  # Disable for compatibility
            "lora_rank": 64,  # Reduced LoRA rank
            "network_alpha": 64
        }
        print("⚠️  Using ultra-conservative memory configuration for 40GB GPU")
    elif gpu_memory_gb < 60:
        # For 60GB+ GPU
        config = {
            "cond_size": 384,
            "noise_size": 768,
            "num_epochs": 30,
            "validation_steps": 15,
            "checkpointing_steps": 15,
            "test_h": 768,
            "test_w": 768,
            "gradient_checkpointing": True,
            "lora_rank": 96,
            "network_alpha": 96
        }
        print("✅ Using moderate memory configuration")
    else:
        # For 80GB+ GPU
        config = {
            "cond_size": 512,
            "noise_size": 1024,
            "num_epochs": 50,
            "validation_steps": 20,
            "checkpointing_steps": 20,
            "test_h": 1024,
            "test_w": 1024,
            "gradient_checkpointing": False,
            "lora_rank": 128,
            "network_alpha": 128
        }
        print("✅ Using standard memory configuration")
    
    return config

def create_optimized_training_script(config, num_samples):
    """Create memory-optimized training script."""
    
    script_content = f'''#!/bin/bash

# Memory-Optimized BezierAdapter Training Script
# Auto-generated based on available GPU memory

export MODEL_DIR="black-forest-labs/FLUX.1-dev"
export OUTPUT_DIR="./models/bezier_adapter_model_optimized"
export CONFIG="./single_gpu_config.yaml"
export TRAIN_DATA="./examples/bezier_clean.jsonl"
export LOG_PATH="$OUTPUT_DIR/log"

# Enhanced memory management
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export TORCH_CUDNN_SDPA_ENABLED=0
export PYTORCH_DISABLE_CUDNN_SDPA=1

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_PATH

echo "Starting Memory-Optimized BezierAdapter training..."
echo "Configuration:"
echo "  Condition size: {config['cond_size']}"
echo "  Noise size: {config['noise_size']}"
echo "  Epochs: {config['num_epochs']}"
echo "  Training samples: {num_samples}"
echo "  Output: $OUTPUT_DIR"

# Clear GPU memory before starting
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Clean training data not found: $TRAIN_DATA"
    echo "Please run: python fix_bezier_training.py"
    exit 1
fi

# Memory-optimized training
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train.py \\
    --pretrained_model_name_or_path $MODEL_DIR \\
    --cond_size={config['cond_size']} \\
    --noise_size={config['noise_size']} \\
    --subject_column="None" \\
    --spatial_column="bezier_curves" \\
    --target_column="target" \\
    --caption_column="caption" \\
    --ranks {config['lora_rank']} \\
    --network_alphas {config['network_alpha']} \\
    --output_dir=$OUTPUT_DIR \\
    --logging_dir=$LOG_PATH \\
    --mixed_precision="bf16" \\
    --train_data_dir=$TRAIN_DATA \\
    --learning_rate=1e-4 \\
    --train_batch_size=1 \\
    --validation_prompt "Traditional Chinese calligraphy character with elegant brushstrokes" \\
    --num_train_epochs={config['num_epochs']} \\
    --validation_steps={config['validation_steps']} \\
    --checkpointing_steps={config['checkpointing_steps']} \\
    --spatial_test_images "None" \\
    --subject_test_images None \\
    --test_h {config['test_h']} \\
    --test_w {config['test_w']} \\
    --num_validation_images=1 \\
    {"--gradient_checkpointing" if config['gradient_checkpointing'] else ""}

echo "Memory-optimized BezierAdapter training completed!"
echo "Model saved to: $OUTPUT_DIR"

# Clean up GPU memory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
'''
    
    script_path = Path("train/train_bezier_optimized.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"✅ Created optimized training script: {script_path}")
    return script_path

def check_memory_before_training():
    """Check GPU memory and provide recommendations."""
    print("\\n=== Memory Check ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # Clear any existing memory
    torch.cuda.empty_cache()
    
    memory_info = get_gpu_memory_info()
    print(f"GPU Memory Status:")
    print(f"  Total: {memory_info['total']:.1f}GB")
    print(f"  Free: {memory_info['free']:.1f}GB")
    print(f"  Allocated: {memory_info['allocated']:.1f}GB")
    
    if memory_info['free'] < 10:
        print("⚠️  Warning: Less than 10GB free memory")
        print("   Consider closing other processes or reducing training parameters")
        return False
    elif memory_info['free'] < 20:
        print("⚠️  Moderate memory available - using conservative settings")
        return True
    else:
        print("✅ Sufficient memory available")
        return True

def main():
    """Main function to fix bezier training issues."""
    print("=" * 60)
    print("BezierAdapter Training Fix")
    print("=" * 60)
    
    # Step 1: Validate bezier dataset
    dataset_path = "bezier_curves_output_no_visualization/chinese-calligraphy-dataset"
    valid_files = validate_bezier_files(dataset_path)
    
    if not valid_files:
        print("❌ No valid bezier files found. Cannot proceed with training.")
        return False
    
    # Step 2: Create clean training data (limited samples for memory)
    clean_data_path = Path("train/examples/bezier_clean.jsonl")
    max_samples = min(30, len(valid_files))  # Limit to 30 samples for memory
    num_samples = create_clean_training_data(valid_files, clean_data_path, max_samples)
    
    # Step 3: Check GPU memory
    memory_ok = check_memory_before_training()
    if not memory_ok:
        print("⚠️  Memory constraints detected - using minimal configuration")
    
    # Step 4: Create optimized configuration
    memory_info = get_gpu_memory_info()
    if isinstance(memory_info, dict):
        config = create_memory_optimized_config(memory_info['free'])
    else:
        print("❌ Could not determine GPU memory, using minimal config")
        config = create_memory_optimized_config(10)  # Assume minimal memory
    
    # Step 5: Create optimized training script
    script_path = create_optimized_training_script(config, num_samples)
    
    print("\\n" + "=" * 60)
    print("Fix Complete!")
    print("=" * 60)
    print(f"✅ Clean training data: {clean_data_path} ({num_samples} samples)")
    print(f"✅ Optimized training script: {script_path}")
    print(f"✅ Memory configuration: {config['cond_size']}x{config['noise_size']} resolution")
    
    print("\\nNext steps:")
    print("1. Review the optimized settings above")
    print("2. Run the optimized training:")
    print(f"   cd train && bash {script_path.name}")
    print("3. Monitor GPU memory during training")
    print("4. If training succeeds, gradually increase parameters")
    
    # Additional memory tips
    print("\\nMemory optimization tips:")
    print("- If still OOM, reduce cond_size further (256 → 128)")
    print("- Enable gradient checkpointing if not already enabled")
    print("- Reduce validation frequency to save memory")
    print("- Close other GPU processes before training")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)