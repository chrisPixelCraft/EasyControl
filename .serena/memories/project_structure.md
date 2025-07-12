# EasyControl Project Structure

## Root Directory
- `README.md`: Comprehensive project documentation
- `requirements.txt`: Python dependencies
- `app.py`: Gradio web application for interactive usage
- `get_dataset.sh`: Script to download training datasets
- `LICENSE`: Apache 2.0 license

## Core Source Code (`src/`)
- `pipeline.py`: Main FluxPipeline implementation
- `transformer_flux.py`: FluxTransformer2DModel definition
- `lora_helper.py`: LoRA utilities (set_single_lora, set_multi_lora)
- `layers_cache.py`: Custom attention layers with KV caching
- `__init__.py`: Package initialization

## Inference Scripts
- `infer.py`: Basic single-condition inference example
- `infer_multi.py`: Multi-condition inference example
- `infer.ipynb`: Jupyter notebook for interactive inference

## Training Code (`train/`)
- `train.py`: Main training script
- `default_config.yaml`: Accelerate configuration
- `train_spatial.sh`: Spatial condition training script
- `train_subject.sh`: Subject condition training script
- `train_style.sh`: Style condition training script
- `readme.md`: Training documentation

## Training Source (`train/src/`)
- `jsonl_datasets.py`: Dataset loading and preprocessing
- `layers.py`: Training-specific layer implementations
- `pipeline.py`: Training pipeline (variant of main pipeline)
- `prompt_helper.py`: Text encoding utilities
- `transformer_flux.py`: Training transformer model

## Training Examples (`train/examples/`)
- `pose.jsonl`, `subject.jsonl`, `style.jsonl`: Sample training data
- `openpose_data/`, `subject_data/`, `style_data/`: Sample images

## Bézier Curve Extraction
- `bezier_extraction.py`: Main extraction script for datasets
- `bezier_extraction_single_character.py`: Single character processing
- `b_curve_extract_readme.md`: Detailed documentation

## Assets & Testing
- `assets/`: Project images, examples, and documentation assets
- `test_imgs/`: Test images for inference
- `.github/`: GitHub workflows and configurations
- `.serena/`: Serena memory bank files