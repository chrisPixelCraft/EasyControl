# EasyControl Training & Inference Guide

Complete guide for training and inference with EasyControl, including the new BezierAdapter framework for Chinese calligraphy generation.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Run the setup script
bash setup_environment.sh

# Or manually set environment variables
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDNN_SDPA_ENABLED=0
export PYTORCH_DISABLE_CUDNN_SDPA=1
```

### 2. Download Models
```bash
# Download base model (required)
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='black-forest-labs/FLUX.1-dev', local_dir='./FLUX.1-dev')
"

# Download LoRA models (optional but recommended)
python download_ckpt.py
```

### 3. Quick Test
```bash
# Test all inference modes
python test_inference.py

# Run examples
python quick_start_guide.py
```

## 🎯 Training

### Setup Training Configuration
```bash
# Auto-configure for your GPU setup
python configure_training.py
```

### Training Data Preparation
```bash
# Prepare training data (including BezierAdapter)
python prepare_training_data.py
```

### Run Training

#### Spatial Conditions (Canny, Depth, Pose, etc.)
```bash
cd train
bash train_spatial.sh
```

#### Subject Conditions
```bash
cd train
bash train_subject.sh
```

#### Style Conditions
```bash
cd train
bash train_style.sh
```

#### BezierAdapter (New!)
```bash
cd train
bash train_bezier.sh
```

### Training Configuration

**Default Settings:**
- **LoRA Rank**: 128
- **Network Alpha**: 128  
- **Learning Rate**: 1e-4
- **Batch Size**: 1 (required due to multi-resolution)
- **Mixed Precision**: bf16
- **Condition Size**: 512
- **Noise Size**: 1024

**Hardware Requirements:**
- **Minimum**: 1x GPU with 12GB+ VRAM
- **Recommended**: 1x A100/H100 with 40GB+ VRAM
- **Multi-GPU**: Supports 2-4 GPUs with automatic configuration

## 🎨 Inference

### 1. Single Condition Generation

```python
from enhanced_inference import EasyControlInference

# Initialize
inferencer = EasyControlInference()

# Generate with Canny edge control
image = inferencer.single_condition_generate(
    prompt="A futuristic sports car in a cyberpunk city",
    control_type="canny",
    control_image="test_imgs/canny.png",
    height=768,
    width=1024,
    seed=42
)

image.save("output_single.png")
```

**Available Control Types:**
- `canny` - Edge detection control
- `depth` - Depth map control
- `pose` - Human pose control
- `hedsketch` - Sketch control
- `seg` - Segmentation control
- `inpainting` - Inpainting control
- `subject` - Subject consistency
- `ghibli` - Ghibli style transfer

### 2. Multi-Condition Generation

```python
# Combine subject + spatial control
image = inferencer.multi_condition_generate(
    prompt="A SKS person driving a car on a mountain road",
    control_types=["subject", "canny"],  # Subject MUST be first
    control_images=["test_imgs/subject.png", "test_imgs/canny.png"],
    height=768,
    width=1024,
    lora_weights=[1.0, 0.8]  # Different weights for each control
)
```

**Important:** When using multi-condition, subject LoRA must come before spatial LoRA.

### 3. BezierAdapter Generation (New!)

```python
# Generate Chinese calligraphy with Bézier curve guidance
image = inferencer.bezier_guided_generate(
    prompt="Traditional Chinese calligraphy with elegant brushstrokes",
    bezier_curves="bezier_curves_output_no_visualization/chinese-calligraphy-dataset/吉/103535_bezier.json",
    height=1024,
    width=1024,
    seed=42
)
```

### 4. Command Line Interface

```bash
# Single condition
python enhanced_inference.py --mode single --control_type canny --control_image test_imgs/canny.png --prompt "A sports car" --output result.png

# Multi-condition
python enhanced_inference.py --mode multi --control_types subject canny --control_images test_imgs/subject.png test_imgs/canny.png --prompt "A SKS person with a car"

# BezierAdapter
python enhanced_inference.py --mode bezier --bezier_file bezier_curves_output_no_visualization/chinese-calligraphy-dataset/吉/103535_bezier.json --prompt "Chinese calligraphy"
```

## 🎭 BezierAdapter Framework

The BezierAdapter is a new addition that enables Bézier curve-guided font stylization for Chinese calligraphy generation.

### Features
- **Parameter Efficient**: ~24.6M parameters (93% reduction vs ControlNet)
- **5 Core Modules**: BezierParameterProcessor, ConditionInjectionAdapter, SpatialAttentionFuser, DensityAdaptiveSampler, StyleBezierFusionModule
- **Real Dataset**: Trained on extracted Bézier curves from Chinese calligraphy dataset

### Dataset Preparation
```bash
# Extract Bézier curves from calligraphy images
python bezier_extraction.py

# This creates bezier_curves_output_no_visualization/ with JSON files
```

### BezierAdapter Training
```bash
cd train
bash train_bezier.sh
```

### BezierAdapter Inference
```python
# Load bezier curves from dataset
from src.bezier_adapter.data_utils import load_bezier_curves_from_dataset

bezier_curves = load_bezier_curves_from_dataset("path/to/bezier/file.json")

# Generate with bezier guidance
image = inferencer.bezier_guided_generate(
    prompt="Elegant Chinese calligraphy character",
    bezier_curves=bezier_curves,
    height=1024,
    width=1024
)
```

## 🌐 Web Interface

### Gradio Interface
```bash
# Start web interface
python app.py
```

Features:
- Interactive parameter adjustment
- Real-time generation
- Multi-condition support
- BezierAdapter integration
- Example gallery

## 🔧 Advanced Configuration

### Memory Optimization
```python
# For limited VRAM
inferencer = EasyControlInference(device="cuda")

# Use smaller resolutions
image = inferencer.single_condition_generate(
    # ... other params
    height=512,  # Instead of 768
    width=512,   # Instead of 1024
    num_inference_steps=15  # Instead of 25
)

# Always clear cache
inferencer.clear_cache()
```

### Custom LoRA Weights
```python
# Fine-tune control strength
image = inferencer.single_condition_generate(
    # ... other params
    lora_weight=0.8  # Reduce control strength
)

# Multi-condition with different weights
image = inferencer.multi_condition_generate(
    # ... other params
    lora_weights=[1.2, 0.6]  # Strong subject, weak spatial
)
```

### Generation Parameters
- **guidance_scale**: 3.5 (default), higher = more prompt adherence
- **num_inference_steps**: 25 (default), higher = better quality
- **height/width**: 768x1024 (default), multiples of 64
- **seed**: Random seed for reproducible results

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce resolution and steps
python enhanced_inference.py --height 512 --width 512 --num_steps 15
```

**2. cuDNN Errors**
```bash
# Set environment variables
export TORCH_CUDNN_SDPA_ENABLED=0
export PYTORCH_DISABLE_CUDNN_SDPA=1
```

**3. Import Errors**
```bash
# Ensure src is in Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

**4. Model Not Found**
```bash
# Download missing models
python download_ckpt.py
```

### Memory Management
```python
# Clear cache after each generation
inferencer.clear_cache()

# Clear GPU memory
torch.cuda.empty_cache()
```

### Training Issues
```bash
# Check training data
python prepare_training_data.py

# Validate environment
python configure_training.py

# Check logs
tail -f train/outputs/logs/training.log
```

## 📊 Performance Benchmarks

### Model Sizes
- **Base FLUX.1-dev**: ~12GB
- **LoRA Models**: ~100-200MB each
- **BezierAdapter**: ~24.6M parameters

### Generation Speed (A100)
- **Single Condition**: ~8-12 seconds (25 steps, 768x1024)
- **Multi-Condition**: ~12-18 seconds  
- **BezierAdapter**: ~10-15 seconds

### Memory Usage
- **Single Generation**: 8-12GB VRAM
- **Multi-Condition**: 10-15GB VRAM
- **Training**: 20-40GB VRAM

## 🎓 Best Practices

### Prompt Engineering
```python
# Good prompts are specific and descriptive
good_prompt = "A sleek red sports car with chrome details driving on a winding mountain road at sunset, professional photography"

# Avoid generic prompts
bad_prompt = "A car"
```

### Control Image Quality
- **Resolution**: Match or exceed output resolution
- **Quality**: High contrast for better control
- **Alignment**: Ensure control matches desired output

### Multi-Condition Order
```python
# CORRECT: Subject first, then spatial
control_types = ["subject", "canny"]

# INCORRECT: Will not work properly
control_types = ["canny", "subject"]
```

### BezierAdapter Tips
- Use traditional calligraphy prompts
- Higher resolution (1024x1024) works best
- Multiple curves create more complex characters

## 📚 API Reference

See the comprehensive API documentation in the code docstrings:
- `enhanced_inference.py` - Main inference class
- `src/bezier_adapter/` - BezierAdapter modules
- `train/train.py` - Training script

## 🤝 Contributing

1. Follow the established code style
2. Add tests for new features
3. Update documentation
4. Test with multiple GPU configurations

## 📝 License

This project follows the EasyControl license terms. See LICENSE file for details.

---

## 🎉 Quick Examples

**Generate a cyberpunk car:**
```bash
python enhanced_inference.py --mode single --control_type canny --control_image test_imgs/canny.png --prompt "A neon-lit cyberpunk car in a futuristic city" --output cyberpunk_car.png
```

**Create subject+scene combination:**
```bash
python enhanced_inference.py --mode multi --control_types subject canny --control_images test_imgs/subject.png test_imgs/canny.png --prompt "A SKS person standing next to a vintage motorcycle" --output subject_scene.png
```

**Generate Chinese calligraphy:**
```bash
python enhanced_inference.py --mode bezier --bezier_file "bezier_curves_output_no_visualization/chinese-calligraphy-dataset/吉/103535_bezier.json" --prompt "Traditional Chinese calligraphy character with flowing brushstrokes" --output calligraphy.png
```

**Start web interface:**
```bash
python app.py
# Open http://localhost:7860 in your browser
```

Happy generating! 🎨✨