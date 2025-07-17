# EasyControl src/ Directory Structure

This document provides a comprehensive guide to the EasyControl source code structure, explaining each file's purpose and how to use them.

## 📁 Directory Structure

```
src/
├── core/              # Core EasyControl components (stable)
├── bezier/            # BezierAdapter framework (experimental)
├── examples/          # Example scripts and demonstrations
├── utils/             # Utility functions and helpers
├── __init__.py        # Main package initialization
└── attention_fix.py   # Attention mechanism fixes
```

## 🔧 Core Components (`src/core/`)

### Essential EasyControl Infrastructure

#### `pipeline.py` - FluxPipeline
**Purpose**: Main diffusion pipeline for image generation
- **Key Classes**: `FluxPipeline`
- **Features**: 
  - Multi-condition support (spatial, subject, style)
  - LoRA integration
  - KV caching for inference efficiency
- **Usage**:
  ```python
  from src.core import FluxPipeline
  pipe = FluxPipeline.from_pretrained("FLUX.1-dev", torch_dtype=torch.bfloat16)
  ```

#### `transformer_flux.py` - FluxTransformer2DModel
**Purpose**: Core FLUX transformer model with 19 double-stream + 38 single-stream blocks
- **Key Classes**: `FluxTransformer2DModel`
- **Features**:
  - Multi-stream attention processing
  - Condition injection through LoRA
  - Efficient memory management
- **Usage**:
  ```python
  from src.core import FluxTransformer2DModel
  transformer = FluxTransformer2DModel.from_pretrained("FLUX.1-dev", subfolder="transformer")
  ```

#### `lora_helper.py` - LoRA Utilities
**Purpose**: Helper functions for LoRA condition injection
- **Key Functions**: `set_single_lora()`, `set_multi_lora()`
- **Features**:
  - Single and multi-condition LoRA setup
  - Weight management
  - Condition size configuration
- **Usage**:
  ```python
  from src.core import set_single_lora, set_multi_lora
  set_single_lora(transformer, "models/canny.safetensors", lora_weights=[1.0])
  set_multi_lora(transformer, ["models/subject.safetensors", "models/canny.safetensors"])
  ```

#### `layers_cache.py` - KV Caching Layers
**Purpose**: Custom attention layers with KV caching for inference efficiency
- **Key Classes**: Custom attention processors with caching
- **Features**:
  - Causal attention with KV cache
  - Memory-efficient inference
  - Bank-based key-value storage
- **Usage**: Automatically used by FluxTransformer2DModel

## 🎨 BezierAdapter Framework (`src/bezier/`)

### Experimental Chinese Calligraphy Generation

#### `bezier_parameter_processor.py` - Density Map Generation
**Purpose**: Convert Bézier curves to density maps using KDE
- **Key Classes**: `BezierParameterProcessor`
- **Features**:
  - KDE density calculation
  - Batch processing support
  - Adaptive bandwidth selection
- **Parameters**: ~2.1M trainable parameters
- **Usage**:
  ```python
  from src.bezier import create_bezier_processor
  processor = create_bezier_processor(device="cuda")
  density_maps = processor(bezier_data_batch)
  ```

#### `enhanced_lora_adapters.py` - Multi-Modal LoRA
**Purpose**: Enhanced LoRA system for multi-modal conditioning
- **Key Classes**: 
  - `EnhancedMultiSingleStreamBlockLoraProcessor`
  - `EnhancedMultiDoubleStreamBlockLoraProcessor`
- **Features**:
  - Style, Text, and Density conditioning
  - Backward compatible with existing LoRA
  - Branch-specific parameter management
- **Parameters**: ~3.6M additional parameters
- **Usage**:
  ```python
  from src.bezier import EnhancedMultiSingleStreamBlockLoraProcessor, ConditionType
  processor = EnhancedMultiSingleStreamBlockLoraProcessor(
      dim=3072, 
      bezier_condition_types=[ConditionType.STYLE, ConditionType.TEXT]
  )
  ```

#### `spatial_attention_fuser.py` - Density-Aware Attention
**Purpose**: Spatial attention fusion with density modulation
- **Key Classes**: `SpatialAttentionFuser`
- **Features**:
  - Density-modulated attention weights
  - Cross-modal attention mechanisms
  - Spatial feature enhancement
- **Parameters**: ~3.8M trainable parameters
- **Usage**:
  ```python
  from src.bezier import SpatialAttentionFuser
  fuser = SpatialAttentionFuser(feature_dim=3072, spatial_dim=64)
  enhanced_features = fuser(features, density_maps)
  ```

#### `style_bezier_fusion_module.py` - Style Transfer
**Purpose**: Style transfer with AdaIN and cross-modal attention
- **Key Classes**: `StyleBezierFusionModule`
- **Features**:
  - AdaIN normalization
  - Multi-head attention
  - Style-density fusion
- **Parameters**: ~3.8M trainable parameters
- **Usage**:
  ```python
  from src.bezier import StyleBezierFusionModule
  fusion = StyleBezierFusionModule(feature_dim=3072, style_dim=768)
  styled_features = fusion(features, style_vectors, density_maps)
  ```

#### `bezier_adapter_pipeline.py` - BezierAdapter Pipeline
**Purpose**: Unified pipeline for BezierAdapter functionality
- **Key Classes**: `BezierAdapterPipeline`
- **Features**:
  - End-to-end Bézier curve processing
  - Multi-modal condition integration
  - Backward compatible with FluxPipeline
- **Usage**:
  ```python
  from src.bezier import BezierAdapterPipeline
  bezier_pipe = BezierAdapterPipeline.from_pretrained("FLUX.1-dev")
  images = bezier_pipe(prompt, bezier_curves=curves, style_vectors=styles)
  ```

#### `bezier_pipeline_integration.py` - Integration Utilities
**Purpose**: Integration utilities for BezierAdapter with EasyControl
- **Key Functions**: Pipeline integration helpers
- **Features**:
  - Seamless integration with existing workflows
  - Condition preprocessing
  - Parameter validation

## 📚 Examples (`src/examples/`)

### Demonstration Scripts

#### `lora_integration_example.py` - Enhanced LoRA Demo
**Purpose**: Comprehensive demonstration of enhanced LoRA system
- **Features**:
  - Real Chinese calligraphy data processing
  - Multi-modal conditioning demonstration
  - Parameter counting and performance analysis
- **Run**: `python src/examples/lora_integration_example.py`

#### `pipeline_integration_example.py` - Full Pipeline Demo  
**Purpose**: Complete pipeline integration demonstration
- **Features**:
  - End-to-end BezierAdapter pipeline
  - Multi-condition processing
  - Image generation examples
- **Run**: `python src/examples/pipeline_integration_example.py`

#### `spatial_attention_integration_example.py` - Spatial Attention Demo
**Purpose**: Spatial attention fusion demonstration
- **Features**:
  - Density-aware attention mechanisms
  - Spatial feature enhancement
  - Performance benchmarks
- **Run**: `python src/examples/spatial_attention_integration_example.py`

#### `style_bezier_fusion_example.py` - Style Fusion Demo
**Purpose**: Style transfer and fusion demonstration
- **Features**:
  - AdaIN style transfer
  - Multi-head attention mechanisms
  - Style-density fusion
- **Run**: `python src/examples/style_bezier_fusion_example.py`

## 🛠️ Utilities (`src/utils/`)

### Helper Functions and Integration Tools

#### `pipeline_integration_utils.py` - Pipeline Helpers
**Purpose**: Utility functions for pipeline integration
- **Features**:
  - Common integration patterns
  - Parameter validation
  - Error handling utilities

#### `spatial_transformer_integration.py` - Spatial Processing
**Purpose**: Spatial processing utilities
- **Features**:
  - Spatial feature processing
  - Coordinate transformations
  - Spatial attention helpers

#### `style_transformer_integration.py` - Style Processing
**Purpose**: Style processing utilities
- **Features**:
  - Style vector processing
  - Style transfer utilities
  - Style fusion helpers

## 📋 Other Files

#### `__init__.py` - Package Initialization
**Purpose**: Main package initialization and exports
- **Features**:
  - Core component exports
  - BezierAdapter component exports
  - Utility exports

#### `attention_fix.py` - Attention Fixes
**Purpose**: Attention mechanism fixes and improvements
- **Features**:
  - cuDNN compatibility fixes
  - Memory optimization
  - Attention mechanism patches

## 🚀 Quick Start Guide

### 1. Standard EasyControl Usage
```bash
# Basic inference
python infer.py

# Multi-condition inference  
python infer_multi.py

# Web interface
python app.py
```

### 2. BezierAdapter Integration
```bash
# Test BezierAdapter components
python src/examples/lora_integration_example.py

# Full pipeline demonstration
python src/examples/pipeline_integration_example.py
```

### 3. Training
```bash
cd train
# Spatial condition training
bash train_spatial.sh

# Subject condition training
bash train_subject.sh

# Style condition training
bash train_style.sh
```

### 4. Testing
```bash
# Test environment and cuDNN
python test_fix.py

# Test BezierAdapter integration
python test_bezier_integration.py

# Test pipeline integration
python test_pipeline_integration.py
```

## 📊 Parameter Breakdown

| Component | Parameters | Purpose |
|-----------|------------|---------|
| **Core EasyControl** | Base Model | FLUX.1-dev diffusion transformer |
| **BezierParameterProcessor** | ~2.1M | Bézier curve to density map conversion |
| **Enhanced LoRA (Single)** | ~1.8M | Multi-modal conditioning (later blocks) |
| **Enhanced LoRA (Double)** | ~1.8M | Multi-modal conditioning (early blocks) |
| **SpatialAttentionFuser** | ~3.8M | Density-aware attention mechanisms |
| **StyleBezierFusionModule** | ~3.8M | Style transfer with AdaIN |
| **Total BezierAdapter** | ~13.3M | Complete BezierAdapter framework |

## 🔧 Environment Setup

### Prerequisites
```bash
# Create conda environment
conda create -n easycontrol python=3.10
conda activate easycontrol

# Install dependencies
pip install -r requirements.txt

# Install system dependencies
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0
```

### Model Download
```bash
# Download all models
bash get_dataset.sh

# Or download specific models
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Xiaojiu-Z/EasyControl', filename='models/canny.safetensors', local_dir='./')"
```

### Training Environment
```bash
# Set cuDNN environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDNN_SDPA_ENABLED=0
export PYTORCH_DISABLE_CUDNN_SDPA=1
```

## 🎯 Usage Patterns

### Core EasyControl
```python
# Standard pattern
from src.core import FluxPipeline, FluxTransformer2DModel, set_single_lora

device = "cuda"
pipe = FluxPipeline.from_pretrained("FLUX.1-dev", torch_dtype=torch.bfloat16)
transformer = FluxTransformer2DModel.from_pretrained("FLUX.1-dev", subfolder="transformer")
pipe.transformer = transformer
set_single_lora(pipe.transformer, "models/canny.safetensors", lora_weights=[1.0])
```

### BezierAdapter
```python
# BezierAdapter pattern
from src.bezier import create_bezier_processor, EnhancedMultiSingleStreamBlockLoraProcessor

# Process Bézier curves
processor = create_bezier_processor(device="cuda")
density_maps = processor(bezier_data)

# Enhanced LoRA with multi-modal conditioning
lora_processor = EnhancedMultiSingleStreamBlockLoraProcessor(
    dim=3072,
    bezier_condition_types=[ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY]
)
```

## 📝 Notes

- **File Preservation**: All original files are preserved in the reorganized structure
- **Backward Compatibility**: Core EasyControl functionality remains unchanged
- **Import Updates**: Some imports may need updating after reorganization
- **Testing Required**: All functionality should be tested after reorganization
- **Documentation**: This README provides comprehensive guidance for all components

## 🔗 Related Files

- `CLAUDE.md` - Project instructions and development patterns
- `Rule.md` - Code review and development rules
- `requirements.txt` - Python dependencies
- `get_dataset.sh` - Model download script
- `train/` - Training scripts and configurations
- `test_*.py` - Testing scripts