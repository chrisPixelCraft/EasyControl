# CLAUDE.md
@Rule.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
EasyControl is an efficient and flexible unified conditional diffusion transformer (DiT) framework for controlled image generation. It enables high-quality image generation with various control inputs including spatial conditions (canny, depth, pose, etc.), subject conditions, and style conditions using lightweight LoRA adapters. The project has been extended with an experimental BezierAdapter framework for Chinese calligraphy generation.

## Essential Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n easycontrol python=3.10
conda activate easycontrol
pip install -r requirements.txt

# Install libGL for OpenCV
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Model Download
```bash
# Download all models at once
bash get_dataset.sh

# Or download specific models
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Xiaojiu-Z/EasyControl', filename='models/canny.safetensors', local_dir='./')"
```

### Inference
```bash
# Single condition inference
python infer.py

# Multi-condition inference
python infer_multi.py

# Interactive web interface (Gradio)
python app.py

# Jupyter notebook
jupyter notebook infer.ipynb
```

### Training
```bash
cd train
# Spatial condition training
bash train_spatial.sh
# Subject condition training
bash train_subject.sh
# Style condition training
bash train_style.sh
```

### Testing
```bash
# Test environment and cuDNN fix
python test_fix.py

# Test BezierAdapter components
python test_bezier_integration.py
python test_pipeline_integration.py
python test_spatial_attention_integration.py
python test_style_bezier_fusion_integration.py
python test_enhanced_lora_system.py

# Run BezierAdapter examples
python src/examples/lora_integration_example.py
python src/examples/pipeline_integration_example.py
python src/examples/spatial_attention_integration_example.py
python src/examples/style_bezier_fusion_example.py
```

## Core Architecture

### Reorganized src/ Directory Structure
The codebase has been reorganized into a modular architecture:
- **src/core/**: Core EasyControl components (stable production code)
  - `pipeline.py` - Main FluxPipeline implementation
  - `transformer_flux.py` - FluxTransformer2DModel (19 double-stream + 38 single-stream blocks)
  - `lora_helper.py` - LoRA utilities (`set_single_lora()`, `set_multi_lora()`)
  - `layers_cache.py` - Custom attention layers with KV caching
- **src/bezier/**: Experimental BezierAdapter framework (~13.3M parameters)
  - `bezier_parameter_processor.py` - Bézier curves to density maps (~2.1M parameters)
  - `enhanced_lora_adapters.py` - Multi-modal LoRA system (~3.6M parameters)
  - `spatial_attention_fuser.py` - Density-aware attention (~3.8M parameters)
  - `style_bezier_fusion_module.py` - Style transfer with AdaIN (~3.8M parameters)
  - `bezier_adapter_pipeline.py` - Unified BezierAdapter pipeline
- **src/examples/**: Demonstration scripts for each component
- **src/utils/**: Integration utilities and helper functions

### Training vs Inference Architecture
The project has **separate implementations** for training and inference:
- **Inference**: Uses `src/` directory with custom pipeline and transformer implementations
- **Training**: Uses `train/src/` directory with training-specific versions of the same components
- **Critical**: Training and inference use different file structures but similar APIs

### Key Components
1. **Base Model**: FLUX.1-dev diffusion transformer (downloaded separately)
2. **LoRA Adapters**: Lightweight condition injection modules (.safetensors files in models/)
3. **KV Cache**: Causal attention mechanism for improved inference speed
4. **Multi-condition Support**: Combine spatial and subject conditions with specific ordering
5. **BezierAdapter Framework**: Experimental Chinese calligraphy generation system

## Development Patterns

### Model Initialization (Standard Pattern)
```python
from src.core.pipeline import FluxPipeline
from src.core.transformer_flux import FluxTransformer2DModel
from src.core.lora_helper import set_single_lora, set_multi_lora

# Standard initialization pattern
device = "cuda"
base_path = "FLUX.1-dev"  # or "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
transformer = FluxTransformer2DModel.from_pretrained(base_path, subfolder="transformer", torch_dtype=torch.bfloat16, device=device)
pipe.transformer = transformer
pipe.to(device)

# Clear cache function (always define this)
def clear_cache(transformer):
    for name, attn_processor in transformer.attn_processors.items():
        attn_processor.bank_kv.clear()
```

### BezierAdapter Integration Pattern
```python
from src.bezier.bezier_parameter_processor import create_bezier_processor
from src.bezier.enhanced_lora_adapters import EnhancedMultiSingleStreamBlockLoraProcessor
from src.bezier.spatial_attention_fuser import SpatialAttentionFuser
from src.bezier.style_bezier_fusion_module import StyleBezierFusionModule

# BezierAdapter components initialization
bezier_processor = create_bezier_processor(device="cuda")
density_maps = bezier_processor(bezier_data_batch)

# Enhanced LoRA with multi-modal conditioning
lora_processor = EnhancedMultiSingleStreamBlockLoraProcessor(
    dim=3072,
    bezier_condition_types=[ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY]
)

# Spatial attention fusion
spatial_fuser = SpatialAttentionFuser(feature_dim=3072, spatial_dim=64)
enhanced_features = spatial_fuser(features, density_maps)

# Style transfer with AdaIN
style_fusion = StyleBezierFusionModule(feature_dim=3072, style_dim=768)
styled_features = style_fusion(features, style_vectors, density_maps)
```

### LoRA Configuration
- **Single condition**: `set_single_lora(pipe.transformer, path, lora_weights=[1], cond_size=512)`
- **Multi-condition**: `set_multi_lora(pipe.transformer, paths, lora_weights=[[1], [1]], cond_size=512)`
- **Critical Order**: Subject LoRA path must come before spatial LoRA path in multi-condition setup

### Environment Variables for Training
```bash
# Required for training to avoid cuDNN errors
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDNN_SDPA_ENABLED=0
export PYTORCH_DISABLE_CUDNN_SDPA=1
```

### Memory Management
- **Always call** `clear_cache(pipe.transformer)` after each generation
- Use `torch.bfloat16` for memory efficiency
- Batch size limited to 1 for training due to multi-resolution images
- Clear GPU cache before loading models: `torch.cuda.empty_cache()`

### Default Parameters
- **Guidance Scale**: 3.5 (standard starting point)
- **Inference Steps**: 25 (quality/speed balance)
- **Max Sequence Length**: 512 for prompts
- **Condition Size**: 512 (cond_size parameter)
- **LoRA Ranks**: 128 (training configuration)
- **Network Alphas**: 128 (training configuration)

### Gradio Web Interface
The project includes a comprehensive Gradio web interface (`app.py`) with:
- **Single Condition Generation**: Individual control type selection
- **Multi-condition Generation**: Combined subject and spatial conditions
- **Style LoRA Integration**: Additional style transfer options
- **Real-time Parameter Adjustment**: Height, width, seed, guidance scale
- **Interactive Examples**: Pre-configured demonstration cases

## File Organization

### Core Directories
- **src/core/**: Core EasyControl components (stable production code)
- **src/bezier/**: Experimental BezierAdapter framework
- **src/examples/**: Demonstration scripts for each component
- **src/utils/**: Integration utilities and helper functions
- **train/**: Training scripts and training-specific implementations
- **models/**: LoRA model files (.safetensors)
- **test_imgs/**: Sample images for testing
- **FLUX.1-dev/**: Base model directory (downloaded separately)

### Training Data Structure
- **train/examples/**: JSONL files with training data format
- **train/default_config.yaml**: Accelerate configuration for multi-GPU training (4 GPUs)
- **train/train.py**: Main training script with position-aware paradigm

### Control Types
Available LoRA models in models/:
- **Spatial**: canny, depth, hedsketch, pose, seg, inpainting
- **Subject**: subject.safetensors
- **Style**: Ghibli.safetensors (additional style LoRAs available via Shakker-Labs)

### BezierAdapter Parameter Breakdown
| Component | Parameters | Purpose |
|-----------|------------|---------|
| BezierParameterProcessor | ~2.1M | Bézier curve to density map conversion |
| Enhanced LoRA (Single) | ~1.8M | Multi-modal conditioning (later blocks) |
| Enhanced LoRA (Double) | ~1.8M | Multi-modal conditioning (early blocks) |
| SpatialAttentionFuser | ~3.8M | Density-aware attention mechanisms |
| StyleBezierFusionModule | ~3.8M | Style transfer with AdaIN |
| **Total BezierAdapter** | ~13.3M | Complete experimental framework |

## Hardware Requirements
- **Training**: At least 1x NVIDIA H100/H800/A100 (~80GB GPU memory)
- **Inference**: CUDA-compatible GPU (lower memory requirements)

## Critical Implementation Notes
- The project uses **custom transformer and pipeline implementations**, not standard diffusers
- Always use absolute paths for model loading
- Clear KV cache after each generation to prevent memory leaks
- Multi-condition requires specific ordering: subject before spatial conditions
- Training uses position-aware paradigm for multi-resolution support
- cuDNN environment variables are required for training stability
- **Import path changes**: Use `src.core.*` for stable components, `src.bezier.*` for experimental features
- **BezierAdapter compatibility**: All BezierAdapter components are backward compatible with core EasyControl
- **Memory optimization**: BezierAdapter adds ~13.3M parameters - monitor GPU memory usage
- **Testing framework**: Comprehensive test suite available for all components
- **Gradio interface**: Use `python app.py` for interactive web-based generation and testing

## Development Workflow Patterns

### Working with Core EasyControl
1. Use `src.core.*` imports for stable functionality
2. Follow standard initialization pattern with clear cache
3. Test with single condition first, then multi-condition
4. Always clear cache after generation

### Working with BezierAdapter
1. Use `src.bezier.*` imports for experimental features
2. Test individual components with `src/examples/` scripts
3. Use comprehensive test suite for integration validation
4. Monitor parameter count and memory usage

### Training New Models
1. Set cuDNN environment variables before training
2. Use `train/default_config.yaml` for multi-GPU setup
3. Batch size must be 1 for multi-resolution training
4. Monitor training with wandb integration

### Debugging and Testing
1. Run `python test_fix.py` for environment verification
2. Use component-specific test scripts for targeted debugging
3. Check `src/attention_fix.py` for attention mechanism patches
4. Verify imports after any code reorganization