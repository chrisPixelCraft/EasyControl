#!/bin/bash

# EasyControl Environment Setup Script
# This script configures the environment for both training and inference

echo "=== EasyControl Environment Setup ==="

# 1. Set critical cuDNN environment variables to prevent training errors
echo "Setting cuDNN environment variables..."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDNN_SDPA_ENABLED=0
export PYTORCH_DISABLE_CUDNN_SDPA=1

# Make these variables persistent
echo 'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' >> ~/.bashrc
echo 'export TORCH_CUDNN_SDPA_ENABLED=0' >> ~/.bashrc
echo 'export PYTORCH_DISABLE_CUDNN_SDPA=1' >> ~/.bashrc

echo "✅ cuDNN environment variables configured"

# 2. Activate virtual environment
echo "Activating virtual environment..."
if [ -d "venv_linux" ]; then
    source venv_linux/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please create it first:"
    echo "  python -m venv venv_linux"
    echo "  source venv_linux/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 3. Check Python and PyTorch installation
echo "Checking Python and PyTorch..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

if [ $? -eq 0 ]; then
    echo "✅ PyTorch installation verified"
else
    echo "❌ PyTorch installation issue detected"
    exit 1
fi

# 4. Check for required dependencies
echo "Checking required dependencies..."
python -c "
import sys
required_packages = ['accelerate', 'diffusers', 'transformers', 'safetensors', 'torch', 'PIL', 'numpy', 'matplotlib']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f'❌ Missing packages: {missing_packages}')
    print('Please install missing packages with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('✅ All required packages available')
"

# 5. Check GPU availability and memory
echo "Checking GPU resources..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f'GPU {i}: {props.name} ({memory_gb:.1f}GB)')
    
    if torch.cuda.device_count() >= 4:
        print('✅ Multi-GPU training supported (4+ GPUs detected)')
    elif torch.cuda.device_count() >= 1:
        print('⚠️  Single GPU detected. Multi-GPU training config may need adjustment')
    else:
        print('❌ No GPUs detected')
else:
    print('❌ CUDA not available')
"

# 6. Create necessary directories
echo "Creating necessary directories..."
mkdir -p models
mkdir -p logs
mkdir -p outputs
mkdir -p train/outputs
mkdir -p train/logs
echo "✅ Directories created"

# 7. Check for base model
echo "Checking for FLUX.1-dev base model..."
if [ -d "FLUX.1-dev" ]; then
    echo "✅ FLUX.1-dev model found"
else
    echo "⚠️  FLUX.1-dev model not found. You'll need to download it:"
    echo "  Using HuggingFace Hub:"
    echo "  from huggingface_hub import snapshot_download"
    echo "  snapshot_download(repo_id='black-forest-labs/FLUX.1-dev', local_dir='./FLUX.1-dev')"
fi

# 8. Verify BezierAdapter implementation
echo "Verifying BezierAdapter implementation..."
python -c "
import sys
sys.path.append('./src')
try:
    from bezier_adapter import BezierParameterProcessor, StyleBezierFusionModule
    print('✅ BezierAdapter modules available')
except ImportError as e:
    print(f'❌ BezierAdapter import error: {e}')
"

echo ""
echo "=== Environment Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. For training: cd train && bash train_spatial.sh"
echo "2. For inference: python infer.py"
echo "3. For BezierAdapter demo: python examples/bezier_adapter_demo.py"
echo "4. For web interface: python app.py"
echo ""
echo "Environment variables are now set. Restart your terminal or run:"
echo "source ~/.bashrc"