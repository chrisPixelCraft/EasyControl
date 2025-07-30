#!/usr/bin/env python3
"""
Enhanced Inference Script for EasyControl with BezierAdapter Support
Supports single condition, multi-condition, and bezier-guided generation.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from PIL import Image
from typing import List, Optional, Union

# Fix for cuDNN Frontend error with CLIP model
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0"
os.environ["PYTORCH_DISABLE_CUDNN_SDPA"] = "1"

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline import FluxPipeline
from src.transformer_flux import FluxTransformer2DModel
from src.lora_helper import set_single_lora, set_multi_lora, unset_lora

# BezierAdapter imports
from src.bezier_adapter import BezierCurve, BezierConfig
from src.bezier_adapter.data_utils import load_bezier_curves_from_dataset

class EasyControlInference:
    """Enhanced inference class for EasyControl with BezierAdapter support."""
    
    def __init__(self, base_model_path: str = "FLUX.1-dev", device: str = "cuda"):
        """
        Initialize the EasyControl inference pipeline.
        
        Args:
            base_model_path: Path to FLUX.1-dev model
            device: Device to run inference on
        """
        self.device = device
        self.base_model_path = base_model_path
        self.pipe = None
        self.transformer = None
        
        # Model paths
        self.lora_path = "models"
        self.control_models = {
            "canny": f"{self.lora_path}/canny.safetensors",
            "depth": f"{self.lora_path}/depth.safetensors", 
            "hedsketch": f"{self.lora_path}/hedsketch.safetensors",
            "pose": f"{self.lora_path}/pose.safetensors",
            "seg": f"{self.lora_path}/seg.safetensors",
            "inpainting": f"{self.lora_path}/inpainting.safetensors",
            "subject": f"{self.lora_path}/subject.safetensors",
            "ghibli": f"{self.lora_path}/Ghibli.safetensors"
        }
        
        # Initialize pipeline
        self._load_pipeline()
    
    def _load_pipeline(self):
        """Load the FluxPipeline and transformer."""
        print(f"Loading pipeline from {self.base_model_path}")
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load pipeline
        self.pipe = FluxPipeline.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map=None
        )
        
        # Load transformer
        self.transformer = FluxTransformer2DModel.from_pretrained(
            self.base_model_path,
            subfolder="transformer", 
            torch_dtype=torch.bfloat16,
            device_map=None
        )
        
        # Set transformer and move to device
        self.pipe.transformer = self.transformer
        self.pipe.to(self.device)
        
        print(f"✅ Pipeline loaded on {self.device}")
    
    def clear_cache(self):
        """Clear KV cache for memory management."""
        for name, attn_processor in self.transformer.attn_processors.items():
            attn_processor.bank_kv.clear()
    
    def single_condition_generate(
        self,
        prompt: str,
        control_type: str,
        control_image: Union[str, Image.Image],
        height: int = 768,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 25,
        seed: int = 42,
        lora_weight: float = 1.0,
        cond_size: int = 512
    ) -> Image.Image:
        """
        Generate image with single condition control.
        
        Args:
            prompt: Text prompt
            control_type: Type of control (canny, depth, pose, subject, etc.)
            control_image: Control image path or PIL Image
            height: Output height
            width: Output width  
            guidance_scale: Guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed
            lora_weight: LoRA weight
            cond_size: Condition size
            
        Returns:
            Generated PIL Image
        """
        print(f"Single condition generation: {control_type}")
        
        # Load control model
        if control_type not in self.control_models:
            raise ValueError(f"Unknown control type: {control_type}")
        
        lora_path = self.control_models[control_type]
        if not Path(lora_path).exists():
            raise FileNotFoundError(f"LoRA model not found: {lora_path}")
        
        set_single_lora(self.transformer, lora_path, lora_weights=[lora_weight], cond_size=cond_size)
        
        # Prepare control image
        if isinstance(control_image, str):
            control_image = Image.open(control_image).convert("RGB")
        
        # Determine if spatial or subject condition
        spatial_images = []
        subject_images = []
        
        if control_type == "subject":
            subject_images = [control_image]
        else:
            spatial_images = [control_image]
        
        # Generate
        generator = torch.Generator("cpu").manual_seed(seed)
        
        result = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator,
            spatial_images=spatial_images,
            subject_images=subject_images,
            cond_size=cond_size
        )
        
        self.clear_cache()
        return result.images[0]
    
    def multi_condition_generate(
        self,
        prompt: str,
        control_types: List[str],
        control_images: List[Union[str, Image.Image]],
        height: int = 768,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 25,
        seed: int = 42,
        lora_weights: List[float] = None,
        cond_size: int = 512
    ) -> Image.Image:
        """
        Generate image with multiple condition controls.
        
        Args:
            prompt: Text prompt
            control_types: List of control types
            control_images: List of control images
            height: Output height
            width: Output width
            guidance_scale: Guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed
            lora_weights: List of LoRA weights for each control
            cond_size: Condition size
            
        Returns:
            Generated PIL Image
        """
        print(f"Multi-condition generation: {control_types}")
        
        if len(control_types) != len(control_images):
            raise ValueError("Number of control types must match number of control images")
        
        if lora_weights is None:
            lora_weights = [1.0] * len(control_types)
        
        # Prepare LoRA paths (subject must come first)
        lora_paths = []
        subject_images = []
        spatial_images = []
        
        # Separate subject and spatial controls
        for i, control_type in enumerate(control_types):
            if control_type not in self.control_models:
                raise ValueError(f"Unknown control type: {control_type}")
            
            lora_path = self.control_models[control_type]
            if not Path(lora_path).exists():
                raise FileNotFoundError(f"LoRA model not found: {lora_path}")
            
            # Prepare control image
            control_image = control_images[i]
            if isinstance(control_image, str):
                control_image = Image.open(control_image).convert("RGB")
            
            if control_type == "subject":
                subject_images.append(control_image)
                lora_paths.insert(0, lora_path)  # Subject must be first
            else:
                spatial_images.append(control_image)
                lora_paths.append(lora_path)
        
        # Set multi-LoRA (subject first)
        weights_formatted = [[w] for w in lora_weights]
        set_multi_lora(self.transformer, lora_paths, lora_weights=weights_formatted, cond_size=cond_size)
        
        # Generate
        generator = torch.Generator("cpu").manual_seed(seed)
        
        result = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator,
            spatial_images=spatial_images,
            subject_images=subject_images,
            cond_size=cond_size
        )
        
        self.clear_cache()
        return result.images[0]
    
    def bezier_guided_generate(
        self,
        prompt: str,
        bezier_curves: Union[str, List[BezierCurve]],
        height: int = 1024,
        width: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 25,
        seed: int = 42,
        cond_size: int = 512
    ) -> Image.Image:
        """
        Generate image with Bézier curve guidance using BezierAdapter.
        
        Args:
            prompt: Text prompt
            bezier_curves: Bézier curves (file path or list of BezierCurve objects)
            height: Output height
            width: Output width
            guidance_scale: Guidance scale
            num_inference_steps: Number of denoising steps
            seed: Random seed
            cond_size: Condition size
            
        Returns:
            Generated PIL Image
        """
        print("Bézier-guided generation with BezierAdapter")
        
        # Load bezier curves if path is provided
        if isinstance(bezier_curves, str):
            bezier_curves = load_bezier_curves_from_dataset(bezier_curves)
        
        # Generate with bezier guidance
        generator = torch.Generator("cpu").manual_seed(seed)
        
        result = self.pipe(
            prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=512,
            generator=generator,
            bezier_curves=bezier_curves,  # BezierAdapter integration
            spatial_images=[],
            subject_images=[],
            cond_size=cond_size
        )
        
        self.clear_cache()
        return result.images[0]

def main():
    """Main inference function with CLI interface."""
    parser = argparse.ArgumentParser(description="EasyControl Enhanced Inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--mode", type=str, choices=["single", "multi", "bezier"], default="single", help="Inference mode")
    parser.add_argument("--control_type", type=str, help="Control type for single mode")
    parser.add_argument("--control_types", type=str, nargs="+", help="Control types for multi mode")
    parser.add_argument("--control_image", type=str, help="Control image path for single mode")
    parser.add_argument("--control_images", type=str, nargs="+", help="Control image paths for multi mode")
    parser.add_argument("--bezier_file", type=str, help="Bézier curve JSON file for bezier mode")
    parser.add_argument("--height", type=int, default=768, help="Output height")
    parser.add_argument("--width", type=int, default=1024, help="Output width")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--num_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--base_model", type=str, default="FLUX.1-dev", help="Base model path")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    
    args = parser.parse_args()
    
    # Initialize inference
    inferencer = EasyControlInference(base_model_path=args.base_model, device=args.device)
    
    # Generate based on mode
    if args.mode == "single":
        if not args.control_type or not args.control_image:
            raise ValueError("Single mode requires --control_type and --control_image")
        
        image = inferencer.single_condition_generate(
            prompt=args.prompt,
            control_type=args.control_type,
            control_image=args.control_image,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            seed=args.seed
        )
        
    elif args.mode == "multi":
        if not args.control_types or not args.control_images:
            raise ValueError("Multi mode requires --control_types and --control_images")
        
        image = inferencer.multi_condition_generate(
            prompt=args.prompt,
            control_types=args.control_types,
            control_images=args.control_images,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            seed=args.seed
        )
        
    elif args.mode == "bezier":
        if not args.bezier_file:
            raise ValueError("Bezier mode requires --bezier_file")
        
        image = inferencer.bezier_guided_generate(
            prompt=args.prompt,
            bezier_curves=args.bezier_file,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            seed=args.seed
        )
    
    # Save result
    image.save(args.output)
    print(f"✅ Generated image saved to {args.output}")

if __name__ == "__main__":
    main()