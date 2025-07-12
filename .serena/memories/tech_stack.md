# EasyControl Tech Stack

## Core Technologies
- **Python**: 3.10 (recommended)
- **PyTorch**: 2.5.1+cu121 with CUDA support
- **Diffusers**: 0.32.2 (HuggingFace diffusion models library)
- **Transformers**: 4.49.0 (HuggingFace transformers)

## Deep Learning & AI
- **PEFT**: 0.14.0 (Parameter-Efficient Fine-Tuning)
- **SafeTensors**: 0.5.2 (secure tensor serialization)
- **Einops**: 0.8.1 (tensor operations)

## Computer Vision
- **OpenCV**: opencv-python (image processing)
- **Pillow**: 11.0.0 (image manipulation)
- **SciPy**: Scientific computing library
- **NumPy**: Numerical computing

## Web Interface
- **Gradio**: Web-based UI for model interaction
- **Spaces**: 0.34.1 (HuggingFace Spaces integration)

## Training & Monitoring
- **Datasets**: HuggingFace datasets library
- **Weights & Biases**: wandb (experiment tracking)
- **Accelerate**: Multi-GPU training support

## Development Tools
- **Git**: Version control
- **Conda**: Environment management
- **Shell scripts**: Automated training and setup

## Hardware Requirements
- **Training**: At least 1x NVIDIA H100/H800/A100 (~80GB GPU memory)
- **Inference**: CUDA-compatible GPU (lower memory requirements)
- **Storage**: ~10GB+ for models and datasets