#!/usr/bin/env python3
"""
Training Configuration Script for EasyControl
Automatically detects GPU setup and configures training accordingly.
"""

import os
import yaml
import torch
import subprocess
from pathlib import Path

def detect_gpu_setup():
    """Detect available GPUs and recommend configuration."""
    if not torch.cuda.is_available():
        return 0, "No CUDA support detected"
    
    gpu_count = torch.cuda.device_count()
    gpu_info = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        gpu_info.append({
            'id': i,
            'name': props.name,
            'memory_gb': memory_gb
        })
    
    return gpu_count, gpu_info

def create_accelerate_config(gpu_count, output_dir="train"):
    """Create appropriate accelerate configuration based on GPU count."""
    
    if gpu_count == 0:
        print("❌ No GPUs detected. Training requires at least 1 GPU.")
        return False
    
    elif gpu_count == 1:
        config = {
            'compute_environment': 'LOCAL_MACHINE',
            'debug': False,
            'distributed_type': 'NO',
            'main_process_port': 14121,
            'downcast_bf16': 'no',
            'gpu_ids': '0',
            'machine_rank': 0,
            'main_training_function': 'main',
            'mixed_precision': 'bf16',
            'num_machines': 1,
            'num_processes': 1,
            'same_network': True,
            'tpu_env': [],
            'tpu_use_cluster': False,
            'tpu_use_sudo': False,
            'use_cpu': False
        }
        config_name = 'single_gpu_config.yaml'
        
    else:  # Multi-GPU
        config = {
            'compute_environment': 'LOCAL_MACHINE',
            'debug': False,
            'distributed_type': 'MULTI_GPU',
            'main_process_port': 14121,
            'downcast_bf16': 'no',
            'gpu_ids': 'all',
            'machine_rank': 0,
            'main_training_function': 'main',
            'mixed_precision': 'bf16',
            'num_machines': 1,
            'num_processes': min(gpu_count, 4),  # Limit to 4 GPUs for stability
            'same_network': True,
            'tpu_env': [],
            'tpu_use_cluster': False,
            'tpu_use_sudo': False,
            'use_cpu': False
        }
        config_name = 'multi_gpu_config.yaml'
    
    # Write configuration
    config_path = Path(output_dir) / config_name
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created accelerate config: {config_path}")
    return config_path

def update_training_scripts(config_path):
    """Update training scripts to use the correct configuration."""
    train_dir = Path("train")
    
    # Update all training scripts
    for script_name in ["train_spatial.sh", "train_subject.sh", "train_style.sh"]:
        script_path = train_dir / script_name
        if script_path.exists():
            # Read current script
            with open(script_path, 'r') as f:
                content = f.read()
            
            # Update CONFIG path
            content = content.replace(
                'export CONFIG="./default_config.yaml"',
                f'export CONFIG="./{config_path.name}"'
            )
            
            # Write back
            with open(script_path, 'w') as f:
                f.write(content)
            
            print(f"✅ Updated {script_name} to use {config_path.name}")

def create_bezier_training_script():
    """Create specialized training script for BezierAdapter."""
    
    script_content = '''#!/bin/bash

# BezierAdapter Training Script for EasyControl
# Trains the BezierAdapter framework on Chinese calligraphy data

export MODEL_DIR="black-forest-labs/FLUX.1-dev" # your flux path
export OUTPUT_DIR="./models/bezier_adapter_model"  # your save path
export CONFIG="./single_gpu_config.yaml"  # will be updated by configure_training.py
export TRAIN_DATA="./examples/bezier.jsonl" # bezier training data
export LOG_PATH="$OUTPUT_DIR/log"

# Fix for cuDNN Frontend error with CLIP model
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDNN_SDPA_ENABLED=0
export PYTORCH_DISABLE_CUDNN_SDPA=1

# Create output directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_PATH

echo "Starting BezierAdapter training..."
echo "Model: $MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo "Data: $TRAIN_DATA"

# Check if training data exists
if [ ! -f "$TRAIN_DATA" ]; then
    echo "❌ Training data not found: $TRAIN_DATA"
    echo "Please create bezier training data first"
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train.py \\
    --pretrained_model_name_or_path $MODEL_DIR \\
    --cond_size=512 \\
    --noise_size=1024 \\
    --subject_column="None" \\
    --spatial_column="bezier_curves" \\
    --target_column="target" \\
    --caption_column="caption" \\
    --ranks 128 \\
    --network_alphas 128 \\
    --output_dir=$OUTPUT_DIR \\
    --logging_dir=$LOG_PATH \\
    --mixed_precision="bf16" \\
    --train_data_dir=$TRAIN_DATA \\
    --learning_rate=1e-4 \\
    --train_batch_size=1 \\
    --validation_prompt "Traditional Chinese calligraphy character with elegant brushstrokes" \\
    --num_train_epochs=500 \\
    --validation_steps=50 \\
    --checkpointing_steps=50 \\
    --spatial_test_images "../bezier_curves_output_no_visualization/chinese-calligraphy-dataset/吁/103527_bezier.json" \\
    --subject_test_images None \\
    --test_h 1024 \\
    --test_w 1024 \\
    --num_validation_images=2

echo "BezierAdapter training completed!"
echo "Model saved to: $OUTPUT_DIR"
'''
    
    script_path = Path("train/train_bezier.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    print(f"✅ Created BezierAdapter training script: {script_path}")

def main():
    """Main configuration function."""
    print("=== EasyControl Training Configuration ===")
    
    # Detect GPU setup
    gpu_count, gpu_info = detect_gpu_setup()
    
    if gpu_count == 0:
        print("❌ No GPUs detected. Cannot proceed with training setup.")
        return False
    
    print(f"✅ Detected {gpu_count} GPU(s):")
    if isinstance(gpu_info, list):
        for gpu in gpu_info:
            print(f"  GPU {gpu['id']}: {gpu['name']} ({gpu['memory_gb']:.1f}GB)")
            
            # Check minimum memory requirements
            if gpu['memory_gb'] < 12:
                print(f"  ⚠️  GPU {gpu['id']} has less than 12GB memory. Training may be limited.")
    
    # Create accelerate configuration
    config_path = create_accelerate_config(gpu_count)
    if not config_path:
        return False
    
    # Update training scripts
    update_training_scripts(config_path)
    
    # Create BezierAdapter training script
    create_bezier_training_script()
    
    print("\n=== Training Configuration Complete ===")
    print(f"Configuration: {config_path}")
    print("\nAvailable training commands:")
    print("  cd train")
    print("  bash train_spatial.sh     # Train spatial conditions")
    print("  bash train_subject.sh     # Train subject conditions") 
    print("  bash train_style.sh       # Train style conditions")
    print("  bash train_bezier.sh      # Train BezierAdapter")
    
    print("\nNext steps:")
    print("1. Prepare your training data in JSONL format")
    print("2. Download FLUX.1-dev base model if not already available")
    print("3. Run the appropriate training script")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)