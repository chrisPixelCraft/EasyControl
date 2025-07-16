"""
BezierPipelineIntegration: Integration layer between BezierParameterProcessor and EasyControl FluxPipeline.

This module provides the necessary components to integrate Bézier curve density conditioning
into the existing EasyControl FLUX transformer pipeline, maintaining compatibility with
existing spatial and subject conditioning while adding density-guided generation capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import json
import os

from bezier_parameter_processor import BezierParameterProcessor, create_bezier_processor
from pipeline import FluxPipeline

class BezierConditionEncoder(nn.Module):
    """
    Encodes Bézier density maps into the format expected by FLUX transformer.

    This module processes density maps from BezierParameterProcessor and converts them
    into the latent space format compatible with existing EasyControl conditioning.
    """

    def __init__(self,
                 latent_channels: int = 16,  # FLUX latent channels
                 density_channels: int = 1,  # Single channel density maps
                 hidden_dim: int = 256):
        super().__init__()

        self.latent_channels = latent_channels
        self.density_channels = density_channels
        self.hidden_dim = hidden_dim

        # Density-to-latent conversion network
        self.density_encoder = nn.Sequential(
            nn.Conv2d(density_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_channels, kernel_size=1),
            nn.Tanh()  # Ensure output is in [-1, 1] range
        )

        # Density feature extraction (for attention weighting)
        self.density_feature_extractor = nn.Sequential(
            nn.Conv2d(density_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, hidden_dim),
            nn.ReLU()
        )

    def forward(self, density_maps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode density maps into latent space format.

        Args:
            density_maps: Tensor of shape [B, 1, H, W] from BezierParameterProcessor

        Returns:
            Tuple of (encoded_latents, density_features)
            - encoded_latents: [B, latent_channels, H, W]
            - density_features: [B, hidden_dim]
        """
        # Encode density maps to latent space
        encoded_latents = self.density_encoder(density_maps)

        # Extract density features for attention
        density_features = self.density_feature_extractor(density_maps)

        return encoded_latents, density_features

class BezierFluxPipeline(FluxPipeline):
    """
    Extended FluxPipeline with Bézier curve density conditioning capabilities.

    This pipeline maintains all existing functionality while adding support for
    Bézier curve density conditioning through the BezierParameterProcessor.
    """

    def __init__(self,
                 scheduler,
                 vae,
                 text_encoder,
                 tokenizer,
                 text_encoder_2,
                 tokenizer_2,
                 transformer,
                 bezier_processor: Optional[BezierParameterProcessor] = None,
                 enable_bezier_conditioning: bool = True):
        """
        Initialize the Bézier-enabled FLUX pipeline.

        Args:
            scheduler: Diffusion scheduler
            vae: VAE model for image encoding/decoding
            text_encoder: CLIP text encoder
            tokenizer: CLIP tokenizer
            text_encoder_2: T5 text encoder
            tokenizer_2: T5 tokenizer
            transformer: FLUX transformer model
            bezier_processor: BezierParameterProcessor instance
            enable_bezier_conditioning: Whether to enable Bézier conditioning
        """
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer
        )

        self.enable_bezier_conditioning = enable_bezier_conditioning

        if enable_bezier_conditioning:
            # Initialize Bézier processor
            if bezier_processor is None:
                device = next(transformer.parameters()).device
                bezier_processor = create_bezier_processor(device=device)

            self.bezier_processor = bezier_processor

            # Initialize condition encoder
            self.bezier_condition_encoder = BezierConditionEncoder()

            # Move to appropriate device
            device = next(transformer.parameters()).device
            self.bezier_condition_encoder = self.bezier_condition_encoder.to(device)

    def prepare_bezier_latents(self,
                              batch_size: int,
                              num_channels_latents: int,
                              height: int,
                              width: int,
                              dtype: torch.dtype,
                              device: torch.device,
                              generator: torch.Generator,
                              bezier_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                              bezier_json_paths: Optional[List[str]] = None,
                              bezier_scale: float = 1.0) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare Bézier curve density conditioning latents.

        Args:
            batch_size: Batch size
            num_channels_latents: Number of latent channels
            height: Latent height
            width: Latent width
            dtype: Data type
            device: Device
            generator: Random generator
            bezier_data: Bézier curve data (from BezierCurveExtractor)
            bezier_json_paths: List of paths to Bézier JSON files
            bezier_scale: Scale factor for Bézier conditioning strength

        Returns:
            Tuple of (bezier_latents, density_features)
        """
        if not self.enable_bezier_conditioning or (bezier_data is None and bezier_json_paths is None):
            return torch.zeros(batch_size, num_channels_latents, height, width,
                             dtype=dtype, device=device), None

        # Process Bézier data to density maps
        if bezier_data is not None:
            # Direct data processing
            if isinstance(bezier_data, dict):
                bezier_data = [bezier_data] * batch_size
            elif len(bezier_data) == 1 and batch_size > 1:
                bezier_data = bezier_data * batch_size

            density_maps = self.bezier_processor(bezier_data)

        elif bezier_json_paths is not None:
            # Load from JSON files
            bezier_data_list = []
            for json_path in bezier_json_paths:
                with open(json_path, 'r', encoding='utf-8') as f:
                    bezier_data_list.append(json.load(f))

            # Extend to batch size if needed
            if len(bezier_data_list) == 1 and batch_size > 1:
                bezier_data_list = bezier_data_list * batch_size

            density_maps = self.bezier_processor(bezier_data_list)

        # Resize density maps to match latent dimensions
        if density_maps.shape[-2:] != (height, width):
            density_maps = F.interpolate(
                density_maps,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )

        # Encode density maps to latent space
        bezier_latents, density_features = self.bezier_condition_encoder(density_maps)

        # Scale conditioning strength
        bezier_latents = bezier_latents * bezier_scale

        return bezier_latents, density_features

    def prepare_latents(self,
                       batch_size: int,
                       num_channels_latents: int,
                       height: int,
                       width: int,
                       dtype: torch.dtype,
                       device: torch.device,
                       generator: torch.Generator,
                       subject_image: Optional[torch.Tensor] = None,
                       condition_image: Optional[torch.Tensor] = None,
                       latents: Optional[torch.Tensor] = None,
                       cond_number: int = 1,
                       sub_number: int = 1,
                       bezier_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                       bezier_json_paths: Optional[List[str]] = None,
                       bezier_scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extended prepare_latents method with Bézier conditioning support.

        Args:
            batch_size: Batch size
            num_channels_latents: Number of latent channels
            height: Latent height
            width: Latent width
            dtype: Data type
            device: Device
            generator: Random generator
            subject_image: Subject conditioning image
            condition_image: Spatial conditioning image
            latents: Initial latents
            cond_number: Number of spatial conditions
            sub_number: Number of subject conditions
            bezier_data: Bézier curve data
            bezier_json_paths: Paths to Bézier JSON files
            bezier_scale: Bézier conditioning strength

        Returns:
            Tuple of (cond_latents, latent_image_ids, noise_latents)
        """
        # Get original latents from parent class
        cond_latents, latent_image_ids, noise_latents = super().prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            subject_image=subject_image,
            condition_image=condition_image,
            latents=latents,
            cond_number=cond_number,
            sub_number=sub_number
        )

        # Prepare Bézier latents
        bezier_latents, density_features = self.prepare_bezier_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
            bezier_data=bezier_data,
            bezier_json_paths=bezier_json_paths,
            bezier_scale=bezier_scale
        )

        # Combine with existing condition latents
        if torch.any(bezier_latents != 0):
            cond_latents = cond_latents + bezier_latents

        # Store density features for attention (if needed)
        if density_features is not None:
            self.density_features = density_features

        return cond_latents, latent_image_ids, noise_latents

    @torch.no_grad()
    def __call__(self,
                prompt: Union[str, List[str]] = None,
                prompt_2: Optional[Union[str, List[str]]] = None,
                subject_images: Optional[List] = None,
                spatial_images: Optional[List] = None,
                bezier_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
                bezier_json_paths: Optional[List[str]] = None,
                bezier_scale: float = 1.0,
                height: int = 512,
                width: int = 512,
                num_inference_steps: int = 20,
                timesteps: List[int] = None,
                guidance_scale: float = 3.5,
                num_images_per_prompt: int = 1,
                generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                latents: Optional[torch.Tensor] = None,
                prompt_embeds: Optional[torch.Tensor] = None,
                pooled_prompt_embeds: Optional[torch.Tensor] = None,
                output_type: str = "pil",
                return_dict: bool = True,
                joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
                callback_on_step_end_tensor_inputs: List[str] = ["latents"],
                max_sequence_length: int = 512) -> Union[torch.Tensor, np.ndarray]:
        """
        Extended __call__ method with Bézier conditioning support.

        Args:
            prompt: Text prompt(s)
            prompt_2: Secondary text prompt(s)
            subject_images: Subject conditioning images
            spatial_images: Spatial conditioning images
            bezier_data: Bézier curve data (from BezierCurveExtractor)
            bezier_json_paths: Paths to Bézier JSON files
            bezier_scale: Bézier conditioning strength (0.0 to 2.0)
            height: Output height
            width: Output width
            num_inference_steps: Number of denoising steps
            timesteps: Custom timesteps
            guidance_scale: Guidance scale
            num_images_per_prompt: Number of images per prompt
            generator: Random generator
            latents: Initial latents
            prompt_embeds: Pre-computed prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            output_type: Output type
            return_dict: Whether to return dict
            joint_attention_kwargs: Joint attention kwargs
            callback_on_step_end: Callback function
            callback_on_step_end_tensor_inputs: Callback inputs
            max_sequence_length: Maximum sequence length

        Returns:
            Generated images
        """
        # Validate Bézier inputs
        if bezier_data is not None and bezier_json_paths is not None:
            raise ValueError("Cannot specify both bezier_data and bezier_json_paths")

        # Prepare conditioning
        cond_number = len(spatial_images) if spatial_images else 0
        sub_number = len(subject_images) if subject_images else 0

        # Process images
        if sub_number > 0:
            subject_image_ls = []
            for img in subject_images:
                subject_image = self.image_processor.preprocess(img, height=self.cond_size, width=self.cond_size)
                subject_image = subject_image.to(dtype=torch.float32)
                subject_image_ls.append(subject_image)
            subject_image = torch.concat(subject_image_ls, dim=-2)
        else:
            subject_image = None

        if cond_number > 0:
            condition_image_ls = []
            for img in spatial_images:
                condition_image = self.image_processor.preprocess(img, height=self.cond_size, width=self.cond_size)
                condition_image = condition_image.to(dtype=torch.float32)
                condition_image_ls.append(condition_image)
            condition_image = torch.concat(condition_image_ls, dim=-2)
        else:
            condition_image = None

        # Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # Get LoRA scale
        lora_scale = (
            joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None
        )

        # Encode prompts
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # Prepare latents with Bézier conditioning
        num_channels_latents = self.transformer.config.in_channels // 4
        cond_latents, latent_image_ids, noise_latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            subject_image,
            condition_image,
            latents,
            cond_number,
            sub_number,
            bezier_data=bezier_data,
            bezier_json_paths=bezier_json_paths,
            bezier_scale=bezier_scale
        )

        # Continue with standard pipeline
        latents = noise_latents

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents

                # Prepare condition latents
                cond_latent_model_input = torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Transform timestep
                timestep = timestep.to(latents.dtype)

                # predict the noise residual
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    cond_hidden_states=cond_latent_model_input,
                    timestep=timestep,
                    guidance=guidance_scale,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Decode latents to images
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return {"images": image}


def create_bezier_flux_pipeline(
    flux_pipeline: FluxPipeline,
    bezier_processor: Optional[BezierParameterProcessor] = None,
    enable_bezier_conditioning: bool = True) -> BezierFluxPipeline:
    """
    Factory function to create a BezierFluxPipeline from an existing FluxPipeline.

    Args:
        flux_pipeline: Existing FluxPipeline instance
        bezier_processor: Optional BezierParameterProcessor instance
        enable_bezier_conditioning: Whether to enable Bézier conditioning

    Returns:
        BezierFluxPipeline instance
    """
    bezier_pipeline = BezierFluxPipeline(
        scheduler=flux_pipeline.scheduler,
        vae=flux_pipeline.vae,
        text_encoder=flux_pipeline.text_encoder,
        tokenizer=flux_pipeline.tokenizer,
        text_encoder_2=flux_pipeline.text_encoder_2,
        tokenizer_2=flux_pipeline.tokenizer_2,
        transformer=flux_pipeline.transformer,
        bezier_processor=bezier_processor,
        enable_bezier_conditioning=enable_bezier_conditioning
    )

    # Copy any additional attributes
    for attr in ['cond_size', 'image_processor', 'joint_attention_kwargs']:
        if hasattr(flux_pipeline, attr):
            setattr(bezier_pipeline, attr, getattr(flux_pipeline, attr))

    return bezier_pipeline


# Utility functions for retrieval (needed for denoising loop)
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs
):
    """
    Retrieve timesteps for the scheduler.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps

    return timesteps, num_inference_steps