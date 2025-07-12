# EasyControl Suggested Commands

## Environment Setup
```bash
# Create conda environment
conda create -n easycontrol python=3.10
conda activate easycontrol

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation
```bash
# Download dataset
bash get_dataset.sh

# Or download specific models
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Xiaojiu-Z/EasyControl', filename='models/canny.safetensors', local_dir='./')"
```

## Inference Commands
```bash
# Basic inference
python infer.py

# Multi-condition inference
python infer_multi.py

# Interactive web interface
python app.py

# Jupyter notebook
jupyter notebook infer.ipynb
```

## Training Commands
```bash
# Navigate to training directory
cd train

# Spatial condition training
bash train_spatial.sh

# Subject condition training
bash train_subject.sh

# Style condition training
bash train_style.sh

# Manual training with custom parameters
accelerate launch --config_file default_config.yaml train.py [args]
```

## Bézier Curve Extraction
```bash
# Process entire dataset
python bezier_extraction.py

# Process single character
python bezier_extraction_single_character.py [image_path]
```

## Development & Debugging
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# List directory contents
ls -la
find . -name "*.py" | head -10

# Check memory usage
nvidia-smi

# Git operations
git status
git log --oneline -10
```

## Model Management
```bash
# Download all models
for model in canny depth hedsketch pose seg inpainting subject; do
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='Xiaojiu-Z/EasyControl', filename='models/${model}.safetensors', local_dir='./')"
done

# Check model files
ls -la models/
```

## System Information
```bash
# Python version
python --version

# PyTorch version
python -c "import torch; print(torch.__version__)"

# CUDA version
nvcc --version
```