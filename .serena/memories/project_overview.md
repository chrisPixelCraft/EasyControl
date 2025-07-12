# EasyControl Project Overview

## Purpose
EasyControl is an efficient and flexible unified conditional diffusion transformer (DiT) framework for controlled image generation. It enables high-quality image generation with various types of control inputs including spatial conditions (canny, depth, pose, etc.), subject conditions, and style conditions.

## Key Features
- **Lightweight Condition Injection LoRA**: Efficient model adaptation without full retraining
- **Position-Aware Training**: Handles multiple resolutions and aspect ratios
- **Causal Attention + KV Cache**: Improved inference efficiency
- **Multi-condition Support**: Combine multiple control types (spatial + subject)
- **Plug-and-play**: Compatible with FLUX.1-dev base model

## Architecture
- Based on FLUX.1-dev diffusion transformer
- Uses LoRA (Low-Rank Adaptation) for efficient training
- Supports both single and multi-condition generation
- Includes specialized components for different control types

## Applications
- Controlled image generation with spatial inputs (canny, depth, pose, segmentation)
- Subject-driven generation
- Style transfer (e.g., Ghibli-style portraits)
- Multi-modal conditional generation
- Calligraphy analysis and processing (via Bézier curve extraction)