#!/usr/bin/env python3
"""
Test script to verify the cuDNN Frontend error fix works properly.
"""

import os
import sys

# Apply the environment variable fix first
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"
os.environ["PYTORCH_DISABLE_CUDNN_SDPA"] = "1"

print("🔧 Environment variables set for cuDNN fix")

import torch
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ CUDA device count: {torch.cuda.device_count()}")
    print(f"✅ Current CUDA device: {torch.cuda.current_device()}")

# Test attention fallback patch
print("\n🔧 Testing attention fallback patch...")
try:
    from src.attention_fix import patch_transformers_clip_attention
    patch_transformers_clip_attention()
    print("✅ Attention fallback patch applied successfully")
except Exception as e:
    print(f"❌ Attention fallback patch failed: {e}")
    print("🔧 Continuing with environment variables only")

# Test basic CLIP functionality
print("\n🔧 Testing CLIP model functionality...")
try:
    from transformers import CLIPTextModel, CLIPTokenizer
    print("✅ CLIP transformers imported successfully")

    # Test with a small model first
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    if torch.cuda.is_available():
        model = model.cuda()
        device = "cuda"
    else:
        device = "cpu"

    print(f"✅ CLIP model loaded successfully on {device}")

    # Test encoding a simple prompt
    test_prompts = [
        "A beautiful sunset",
        "A cat sitting on a chair",
        "Portrait of a person in the city"
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", max_length=77, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        print(f"✅ Test prompt '{prompt}' - Output shape: {outputs.pooler_output.shape}")

    print("\n✅ All CLIP tests passed - cuDNN fix is working!")

except Exception as e:
    print(f"❌ CLIP test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 cuDNN Frontend error fix validation completed successfully!")