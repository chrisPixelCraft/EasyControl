#!/bin/bash

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

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=512 \
    --noise_size=1024 \
    --subject_column="None" \
    --spatial_column="bezier_curves" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "Traditional Chinese calligraphy character with elegant brushstrokes" \
    --num_train_epochs=500 \
    --validation_steps=50 \
    --checkpointing_steps=50 \
    --spatial_test_images "../bezier_curves_output_no_visualization/chinese-calligraphy-dataset/吁/103527_bezier.json" \
    --subject_test_images None \
    --test_h 1024 \
    --test_w 1024 \
    --num_validation_images=2

echo "BezierAdapter training completed!"
echo "Model saved to: $OUTPUT_DIR"
