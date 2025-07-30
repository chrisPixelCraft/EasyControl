#!/bin/bash

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
echo "  Condition size: 256"
echo "  Noise size: 512"
echo "  Epochs: 20"
echo "  Training samples: 30"
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
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=256 \
    --noise_size=512 \
    --subject_column="None" \
    --spatial_column="bezier_curves" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 64 \
    --network_alphas 64 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "Traditional Chinese calligraphy character with elegant brushstrokes" \
    --num_train_epochs=20 \
    --validation_steps=20 \
    --checkpointing_steps=20 \
    --spatial_test_images "None" \
    --subject_test_images None \
    --test_h 512 \
    --test_w 512 \
    --num_validation_images=1 \
    

echo "Memory-optimized BezierAdapter training completed!"
echo "Model saved to: $OUTPUT_DIR"

# Clean up GPU memory
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"
