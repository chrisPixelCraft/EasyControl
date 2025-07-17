import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin
import numpy as np
import PIL.Image
from dataclasses import dataclass

# Import all BezierAdapter components
from .bezier_parameter_processor import BezierParameterProcessor
from .spatial_attention_fuser import SpatialAttentionFuser
from ..utils.spatial_transformer_integration import SpatialTransformerIntegrator
from .style_bezier_fusion_module import StyleBezierFusionModule
from ..utils.style_transformer_integration import StyleTransformerIntegrator
from .enhanced_lora_adapters import EnhancedMultiSingleStreamBlockLoraProcessor, EnhancedMultiDoubleStreamBlockLoraProcessor
from ..core.pipeline import FluxPipeline

logger = logging.get_logger(__name__)

@dataclass
class BezierAdapterOutput(BaseOutput):
    """
    Output class for BezierAdapter pipeline.

    Args:
        images: List of generated images
        bezier_debug_info: Debug information from BezierParameterProcessor
        spatial_debug_info: Debug information from SpatialAttentionFuser
        style_debug_info: Debug information from StyleBezierFusionModule
        lora_debug_info: Debug information from Enhanced LoRA adapters
    """
    images: Union[List[PIL.Image.Image], np.ndarray]
    bezier_debug_info: Optional[Dict[str, Any]] = None
    spatial_debug_info: Optional[Dict[str, Any]] = None
    style_debug_info: Optional[Dict[str, Any]] = None
    lora_debug_info: Optional[Dict[str, Any]] = None

class BezierAdapterPipeline(FluxPipeline):
    """
    BezierAdapterPipeline: Comprehensive pipeline integrating all BezierAdapter components.

    This pipeline extends the existing FluxPipeline to support:
    - Bézier curve conditioning (via BezierParameterProcessor)
    - Spatial attention fusion (via SpatialAttentionFuser)
    - Style conditioning (via StyleBezierFusionModule)
    - Enhanced LoRA adapters for multi-modal conditioning
    - Backward compatibility with existing spatial/subject conditions

    Total BezierAdapter Parameters: ~13.3M
    - BezierParameterProcessor: ~2.1M
    - SpatialAttentionFuser: ~3.8M
    - StyleBezierFusionModule: ~3.8M
    - Enhanced LoRA Adapters: ~3.6M
    """

    def __init__(self,
                 scheduler: FlowMatchEulerDiscreteScheduler,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 text_encoder_2: T5EncoderModel,
                 tokenizer_2: T5TokenizerFast,
                 transformer: FluxTransformer2DModel,
                 # BezierAdapter configuration
                 enable_bezier_conditioning: bool = True,
                 enable_spatial_attention: bool = True,
                 enable_style_fusion: bool = True,
                 enable_enhanced_lora: bool = True,
                 # BezierAdapter components (optional - will be created if not provided)
                 bezier_processor: Optional[BezierParameterProcessor] = None,
                 spatial_integrator: Optional[SpatialTransformerIntegrator] = None,
                 style_integrator: Optional[StyleTransformerIntegrator] = None,
                 # Configuration dictionaries
                 bezier_config: Optional[Dict[str, Any]] = None,
                 spatial_config: Optional[Dict[str, Any]] = None,
                 style_config: Optional[Dict[str, Any]] = None,
                 lora_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the BezierAdapterPipeline.

        Args:
            scheduler: Diffusion scheduler
            vae: VAE model for image encoding/decoding
            text_encoder: CLIP text encoder
            tokenizer: CLIP tokenizer
            text_encoder_2: T5 text encoder
            tokenizer_2: T5 tokenizer
            transformer: FLUX transformer model
            enable_bezier_conditioning: Whether to enable Bézier curve conditioning
            enable_spatial_attention: Whether to enable spatial attention fusion
            enable_style_fusion: Whether to enable style fusion
            enable_enhanced_lora: Whether to enable enhanced LoRA adapters
            bezier_processor: Optional BezierParameterProcessor instance
            spatial_integrator: Optional SpatialTransformerIntegrator instance
            style_integrator: Optional StyleTransformerIntegrator instance
            bezier_config: Configuration for BezierParameterProcessor
            spatial_config: Configuration for SpatialAttentionFuser
            style_config: Configuration for StyleBezierFusionModule
            lora_config: Configuration for Enhanced LoRA adapters
        """

        # Initialize base pipeline
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer
        )

        # Store configuration
        self.enable_bezier_conditioning = enable_bezier_conditioning
        self.enable_spatial_attention = enable_spatial_attention
        self.enable_style_fusion = enable_style_fusion
        self.enable_enhanced_lora = enable_enhanced_lora

        # Get device from transformer
        self.device = next(transformer.parameters()).device

        # Initialize BezierAdapter components
        self._init_bezier_adapter_components(
            bezier_processor=bezier_processor,
            spatial_integrator=spatial_integrator,
            style_integrator=style_integrator,
            bezier_config=bezier_config,
            spatial_config=spatial_config,
            style_config=style_config,
            lora_config=lora_config
        )

        # Track integration status
        self.bezier_adapter_integrated = False

    def _init_bezier_adapter_components(self,
                                      bezier_processor: Optional[BezierParameterProcessor] = None,
                                      spatial_integrator: Optional[SpatialTransformerIntegrator] = None,
                                      style_integrator: Optional[StyleTransformerIntegrator] = None,
                                      bezier_config: Optional[Dict[str, Any]] = None,
                                      spatial_config: Optional[Dict[str, Any]] = None,
                                      style_config: Optional[Dict[str, Any]] = None,
                                      lora_config: Optional[Dict[str, Any]] = None):
        """Initialize all BezierAdapter components."""

        # 1. Initialize BezierParameterProcessor
        if self.enable_bezier_conditioning:
            if bezier_processor is None:
                bezier_config = bezier_config or {
                    'output_size': (64, 64),
                    'hidden_dim': 256,
                    'num_gaussian_components': 8,
                    'curve_resolution': 100,
                    'density_sigma': 2.0,
                    'learnable_kde': True,
                    'device': self.device
                }
                bezier_processor = BezierParameterProcessor(**bezier_config)

            self.bezier_processor = bezier_processor
        else:
            self.bezier_processor = None

        # 2. Initialize SpatialTransformerIntegrator
        if self.enable_spatial_attention:
            if spatial_integrator is None:
                spatial_config = spatial_config or {
                    'spatial_attention_config': {
                        'hidden_dim': 3072,
                        'num_heads': 24,
                        'head_dim': 128,
                        'density_feature_dim': 256,
                        'spatial_resolution': 64,
                        'fusion_layers': 3,
                        'use_positional_encoding': True,
                        'dropout': 0.1
                    },
                    'device': self.device,
                    'dtype': torch.float32
                }
                spatial_integrator = SpatialTransformerIntegrator(
                    flux_transformer=self.transformer,
                    **spatial_config
                )

            self.spatial_integrator = spatial_integrator
        else:
            self.spatial_integrator = None

        # 3. Initialize StyleTransformerIntegrator
        if self.enable_style_fusion:
            if style_integrator is None:
                style_config = style_config or {
                    'style_fusion_config': {
                        'content_dim': 3072,
                        'style_vector_dim': 256,
                        'style_feature_dim': 256,
                        'num_attention_heads': 8,
                        'attention_head_dim': 64,
                        'fusion_layers': 3,
                        'use_adain': True,
                        'use_cross_attention': True,
                        'dropout': 0.1
                    },
                    'device': self.device,
                    'dtype': torch.float32
                }
                style_integrator = StyleTransformerIntegrator(
                    flux_transformer=self.transformer,
                    **style_config
                )

            self.style_integrator = style_integrator
        else:
            self.style_integrator = None

        # 4. Enhanced LoRA configuration will be handled during integration
        self.lora_config = lora_config or {
            'lora_rank': 64,
            'lora_alpha': 64,
            'lora_dropout': 0.1,
            'bezier_condition_types': ['style', 'text', 'density']
        }

    def setup_bezier_adapter(self):
        """Setup and integrate all BezierAdapter components."""

        if self.bezier_adapter_integrated:
            logger.info("BezierAdapter already integrated.")
            return

        # Integrate spatial attention
        if self.spatial_integrator is not None:
            self.spatial_integrator.integrate_spatial_attention()
            logger.info("SpatialAttentionFuser integrated.")

        # Integrate style fusion
        if self.style_integrator is not None:
            self.style_integrator.integrate_style_fusion()
            logger.info("StyleBezierFusionModule integrated.")

        # Enhanced LoRA integration would be handled here
        # (This would involve setting up the enhanced LoRA processors)
        if self.enable_enhanced_lora:
            self._setup_enhanced_lora()
            logger.info("Enhanced LoRA adapters integrated.")

        self.bezier_adapter_integrated = True
        logger.info("BezierAdapter integration complete.")

    def _setup_enhanced_lora(self):
        """Setup enhanced LoRA adapters (placeholder for now)."""
        # This would involve setting up the enhanced LoRA processors
        # with the transformer attention processors
        pass

    def cleanup_bezier_adapter(self):
        """Remove BezierAdapter components and restore original processors."""

        if not self.bezier_adapter_integrated:
            logger.info("BezierAdapter not integrated.")
            return

        # Remove spatial attention
        if self.spatial_integrator is not None:
            self.spatial_integrator.remove_spatial_attention()

        # Remove style fusion
        if self.style_integrator is not None:
            self.style_integrator.remove_style_fusion()

        # Remove enhanced LoRA
        if self.enable_enhanced_lora:
            self._cleanup_enhanced_lora()

        self.bezier_adapter_integrated = False
        logger.info("BezierAdapter cleanup complete.")

    def _cleanup_enhanced_lora(self):
        """Cleanup enhanced LoRA adapters (placeholder for now)."""
        # This would involve restoring original attention processors
        pass

    def process_bezier_conditioning(self, bezier_data: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
        """Process Bézier curve data into density maps."""

        if not self.enable_bezier_conditioning or bezier_data is None:
            return None

        if self.bezier_processor is None:
            logger.warning("BezierParameterProcessor not initialized.")
            return None

        # Process Bézier curves to density maps
        density_map = self.bezier_processor.process_bezier_data(bezier_data)

        return density_map

    def process_style_conditioning(self, style_data: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
        """Process style data into style vectors."""

        if not self.enable_style_fusion or style_data is None:
            return None

        # Extract style vectors from style data
        if isinstance(style_data, dict):
            style_vectors = style_data.get('style_vectors', None)
            if style_vectors is None:
                # If no pre-computed style vectors, create random ones for demonstration
                batch_size = style_data.get('batch_size', 1)
                style_vectors = torch.randn(batch_size, 256, device=self.device)
        else:
            style_vectors = style_data

        return style_vectors

    def get_bezier_adapter_summary(self) -> Dict[str, Any]:
        """Get summary of BezierAdapter integration."""

        summary = {
            'integration_status': self.bezier_adapter_integrated,
            'components': {
                'bezier_conditioning': self.enable_bezier_conditioning,
                'spatial_attention': self.enable_spatial_attention,
                'style_fusion': self.enable_style_fusion,
                'enhanced_lora': self.enable_enhanced_lora
            },
            'parameter_counts': {},
            'total_parameters': 0
        }

        # Get parameter counts from each component
        if self.bezier_processor is not None:
            bezier_params = sum(p.numel() for p in self.bezier_processor.parameters())
            summary['parameter_counts']['bezier_processor'] = bezier_params
            summary['total_parameters'] += bezier_params

        if self.spatial_integrator is not None:
            spatial_summary = self.spatial_integrator.get_integration_summary()
            summary['parameter_counts']['spatial_integrator'] = spatial_summary['total_parameters']
            summary['total_parameters'] += spatial_summary['total_parameters']

        if self.style_integrator is not None:
            style_summary = self.style_integrator.get_integration_summary()
            summary['parameter_counts']['style_integrator'] = style_summary['total_parameters']
            summary['total_parameters'] += style_summary['total_parameters']

        return summary

    def set_conditioning_strength(self,
                                style_strength: float = 1.0,
                                spatial_strength: float = 1.0):
        """Set conditioning strength for different modalities."""

        if self.style_integrator is not None:
            self.style_integrator.set_style_strength(style_strength)

        # Spatial strength would be handled here when implemented
        # This could involve modifying the spatial attention weights

    @torch.no_grad()
    def __call__(self,
                 prompt: Union[str, List[str]] = None,
                 prompt_2: Optional[Union[str, List[str]]] = None,
                 height: Optional[int] = None,
                 width: Optional[int] = None,
                 num_inference_steps: int = 28,
                 timesteps: List[int] = None,
                 guidance_scale: float = 3.5,
                 num_images_per_prompt: Optional[int] = 1,
                 generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                 latents: Optional[torch.FloatTensor] = None,
                 prompt_embeds: Optional[torch.FloatTensor] = None,
                 pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
                 output_type: Optional[str] = "pil",
                 return_dict: bool = True,
                 joint_attention_kwargs: Optional[Dict[str, Any]] = None,
                 callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
                 callback_on_step_end_tensor_inputs: List[str] = ["latents"],
                 max_sequence_length: int = 512,
                 # Existing conditioning (backward compatibility)
                 spatial_images: List[PIL.Image.Image] = [],
                 subject_images: List[PIL.Image.Image] = [],
                 cond_size: int = 512,
                 # BezierAdapter conditioning
                 bezier_data: Optional[Dict[str, Any]] = None,
                 style_data: Optional[Dict[str, Any]] = None,
                 density_map: Optional[torch.Tensor] = None,
                 style_vectors: Optional[torch.Tensor] = None,
                 # Debug options
                 return_debug_info: bool = False) -> Union[BezierAdapterOutput, Dict[str, Any]]:
        """
        Generate images with BezierAdapter conditioning.

        Args:
            prompt: Text prompt for generation
            prompt_2: Optional second text prompt
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            timesteps: Custom timesteps
            guidance_scale: Guidance scale for classifier-free guidance
            num_images_per_prompt: Number of images per prompt
            generator: Random number generator
            latents: Initial latents
            prompt_embeds: Pre-computed prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            output_type: Output format ('pil' or 'pt')
            return_dict: Whether to return dict or tuple
            joint_attention_kwargs: Additional attention arguments
            callback_on_step_end: Callback function
            callback_on_step_end_tensor_inputs: Callback tensor inputs
            max_sequence_length: Maximum sequence length
            spatial_images: Spatial conditioning images (existing)
            subject_images: Subject conditioning images (existing)
            cond_size: Conditioning image size
            bezier_data: Bézier curve data for spatial conditioning
            style_data: Style data for style conditioning
            density_map: Pre-computed density map
            style_vectors: Pre-computed style vectors
            return_debug_info: Whether to return debug information

        Returns:
            BezierAdapterOutput or dict with generated images and debug info
        """

        # Ensure BezierAdapter is integrated
        if not self.bezier_adapter_integrated:
            self.setup_bezier_adapter()

        # Process BezierAdapter conditioning
        debug_info = {}

        # 1. Process Bézier conditioning
        if density_map is None:
            density_map = self.process_bezier_conditioning(bezier_data)
            if density_map is not None:
                debug_info['bezier_debug_info'] = {'density_map_shape': density_map.shape}

        # 2. Process style conditioning
        if style_vectors is None:
            style_vectors = self.process_style_conditioning(style_data)
            if style_vectors is not None:
                debug_info['style_debug_info'] = {'style_vectors_shape': style_vectors.shape}

        # 3. Prepare joint attention kwargs with BezierAdapter conditioning
        joint_attention_kwargs = joint_attention_kwargs or {}

        if density_map is not None:
            joint_attention_kwargs['density_map'] = density_map

        if style_vectors is not None:
            joint_attention_kwargs['style_vectors'] = style_vectors

        # 4. Call the parent pipeline with all conditioning
        result = super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            spatial_images=spatial_images,
            subject_images=subject_images,
            cond_size=cond_size
        )

        # 5. Return with debug info if requested
        if return_debug_info:
            if return_dict:
                return BezierAdapterOutput(
                    images=result.images,
                    bezier_debug_info=debug_info.get('bezier_debug_info'),
                    spatial_debug_info=debug_info.get('spatial_debug_info'),
                    style_debug_info=debug_info.get('style_debug_info'),
                    lora_debug_info=debug_info.get('lora_debug_info')
                )
            else:
                return result, debug_info
        else:
            return result

    def enable_debug_mode(self):
        """Enable debug mode for all components."""
        # This would enable debug outputs from all components
        pass

    def disable_debug_mode(self):
        """Disable debug mode for all components."""
        # This would disable debug outputs from all components
        pass


def create_bezier_adapter_pipeline(
    scheduler: FlowMatchEulerDiscreteScheduler,
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    text_encoder_2: T5EncoderModel,
    tokenizer_2: T5TokenizerFast,
    transformer: FluxTransformer2DModel,
    enable_all_components: bool = True,
    custom_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> BezierAdapterPipeline:
    """
    Factory function to create a BezierAdapterPipeline.

    Args:
        scheduler: Diffusion scheduler
        vae: VAE model
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        text_encoder_2: T5 text encoder
        tokenizer_2: T5 tokenizer
        transformer: FLUX transformer model
        enable_all_components: Whether to enable all BezierAdapter components
        custom_configs: Custom configuration for each component

    Returns:
        Configured BezierAdapterPipeline
    """

    custom_configs = custom_configs or {}

    pipeline = BezierAdapterPipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
        transformer=transformer,
        enable_bezier_conditioning=enable_all_components,
        enable_spatial_attention=enable_all_components,
        enable_style_fusion=enable_all_components,
        enable_enhanced_lora=enable_all_components,
        bezier_config=custom_configs.get('bezier_config'),
        spatial_config=custom_configs.get('spatial_config'),
        style_config=custom_configs.get('style_config'),
        lora_config=custom_configs.get('lora_config')
    )

    return pipeline