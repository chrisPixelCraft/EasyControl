import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from .style_bezier_fusion_module import StyleBezierFusionModule, StyleBezierProcessor
from .transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock, FluxTransformer2DModel
from diffusers.models.attention_processor import FluxAttnProcessor2_0
import copy

class StyleTransformerIntegrator(nn.Module):
    """
    StyleTransformerIntegrator: Integrates StyleBezierFusionModule with FLUX transformer blocks.

    This module manages the integration of style fusion mechanisms into
    specific transformer blocks according to the architecture mapping from TASK_01:
    - Phase 2: Single-Stream Blocks 5-15 (StyleBezierFusion)
    """

    def __init__(self,
                 flux_transformer: FluxTransformer2DModel,
                 style_fusion_config: Dict[str, Any] = None,
                 integration_phases: Dict[str, List[int]] = None,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the StyleTransformerIntegrator.

        Args:
            flux_transformer: FLUX transformer model to integrate with
            style_fusion_config: Configuration for StyleBezierFusionModule
            integration_phases: Mapping of phases to block indices
            device: Device for computation
            dtype: Data type for computation
        """
        super().__init__()

        self.flux_transformer = flux_transformer
        self.device = device
        self.dtype = dtype

        # Default integration phases based on architecture mapping
        self.integration_phases = integration_phases or {
            'phase2_single_stream': list(range(5, 16))  # Single-stream blocks 5-15
        }

        # Default style fusion configuration
        self.style_fusion_config = style_fusion_config or {
            'content_dim': 3072,
            'style_vector_dim': 256,
            'style_feature_dim': 256,
            'num_attention_heads': 8,
            'attention_head_dim': 64,
            'fusion_layers': 3,
            'use_adain': True,
            'use_cross_attention': True,
            'dropout': 0.1
        }

        # Initialize style fusion modules for different phases
        self._init_style_fusion_modules()

        # Store original attention processors for restoration
        self.original_processors = {}

        # Track integration status
        self.is_integrated = False

    def _init_style_fusion_modules(self):
        """Initialize style fusion modules for different phases."""

        # Phase 2: Single-Stream StyleBezierFusion
        self.phase2_style_fusers = nn.ModuleDict()
        for block_idx in self.integration_phases['phase2_single_stream']:
            fusion_module = StyleBezierFusionModule(
                **self.style_fusion_config,
                device=self.device,
                dtype=self.dtype
            )
            self.phase2_style_fusers[f'block_{block_idx}'] = fusion_module

    def integrate_style_fusion(self):
        """Integrate style fusion mechanisms into the FLUX transformer."""

        if self.is_integrated:
            print("StyleBezierFusionModule already integrated.")
            return

        # Store original processors
        self.original_processors = copy.deepcopy(self.flux_transformer.attn_processors)

        # Get new processors with style fusion integration
        new_processors = {}

        # Phase 2: Single-Stream Blocks Enhancement
        single_stream_start = len(self.flux_transformer.transformer_blocks)
        for block_idx in self.integration_phases['phase2_single_stream']:
            adjusted_idx = block_idx - single_stream_start
            if adjusted_idx < len(self.flux_transformer.single_transformer_blocks):
                block_name = f'single_transformer_blocks.{adjusted_idx}.attn.processor'
                original_processor = self.flux_transformer.attn_processors.get(block_name)

                style_fusion_module = self.phase2_style_fusers[f'block_{block_idx}']
                style_processor = StyleBezierProcessor(
                    style_fusion_module=style_fusion_module,
                    base_processor=original_processor
                )
                new_processors[block_name] = style_processor

        # Update the transformer with new processors
        self.flux_transformer.set_attn_processor(new_processors)
        self.is_integrated = True

        print(f"StyleBezierFusionModule integrated into {len(new_processors)} transformer blocks.")

    def remove_style_fusion(self):
        """Remove style fusion mechanisms and restore original processors."""

        if not self.is_integrated:
            print("StyleBezierFusionModule not integrated.")
            return

        # Restore original processors
        self.flux_transformer.set_attn_processor(self.original_processors)
        self.is_integrated = False

        print("StyleBezierFusionModule removed. Original processors restored.")

    def forward_with_style_conditioning(self,
                                       hidden_states: torch.Tensor,
                                       encoder_hidden_states: torch.Tensor,
                                       timestep: Union[torch.Tensor, float, int],
                                       guidance: torch.Tensor = None,
                                       pooled_projections: torch.Tensor = None,
                                       style_vectors: Optional[torch.Tensor] = None,
                                       return_dict: bool = True,
                                       **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with style conditioning from style vectors.

        Args:
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states
            timestep: Timestep for diffusion
            guidance: Guidance tensor
            pooled_projections: Pooled projections
            style_vectors: Style vectors for conditioning [B, style_vector_dim]
            return_dict: Whether to return dict or tensor
            **kwargs: Additional arguments

        Returns:
            Transformer output with style conditioning
        """

        # Store style vectors in kwargs for processors to access
        if style_vectors is not None:
            kwargs['style_vectors'] = style_vectors

        # Forward pass through the transformer
        return self.flux_transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            return_dict=return_dict,
            **kwargs
        )

    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of style fusion integration."""

        total_params = 0
        phase_params = {}

        # Phase 2 parameters
        phase2_params = sum(p.numel() for fuser in self.phase2_style_fusers.values() for p in fuser.parameters())
        phase_params['phase2_single_stream'] = {
            'blocks': self.integration_phases['phase2_single_stream'],
            'parameters': phase2_params,
            'num_blocks': len(self.integration_phases['phase2_single_stream'])
        }
        total_params += phase2_params

        return {
            'total_parameters': total_params,
            'phase_breakdown': phase_params,
            'integration_status': self.is_integrated,
            'target_parameters': 3800000,  # 3.8M target
            'parameter_efficiency': total_params / 3800000 if total_params > 0 else 0
        }

    def get_detailed_parameter_breakdown(self) -> Dict[str, Any]:
        """Get detailed parameter breakdown for each style fusion module."""

        breakdown = {
            'phase2_single_stream': {}
        }

        # Phase 2 breakdown
        for block_name, fuser in self.phase2_style_fusers.items():
            breakdown['phase2_single_stream'][block_name] = fuser.get_parameter_count()

        return breakdown

    def set_style_strength(self, strength: float):
        """Set style strength for all integrated style fusion modules."""
        for fuser in self.phase2_style_fusers.values():
            fuser.set_style_strength(strength)

    def get_style_strength(self) -> float:
        """Get the current style strength (from first fusion module)."""
        if self.phase2_style_fusers:
            first_fuser = next(iter(self.phase2_style_fusers.values()))
            return first_fuser.get_style_strength()
        return 1.0


class BezierStylePipeline:
    """
    Pipeline for integrating BezierAdapter style conditioning with FLUX transformer.
    """

    def __init__(self,
                 flux_transformer: FluxTransformer2DModel,
                 style_processor: Any = None,  # Style feature processor
                 style_integrator: StyleTransformerIntegrator = None):
        """
        Initialize the BezierStylePipeline.

        Args:
            flux_transformer: FLUX transformer model
            style_processor: Style feature processor instance
            style_integrator: Optional pre-configured StyleTransformerIntegrator
        """
        self.flux_transformer = flux_transformer
        self.style_processor = style_processor

        # Initialize style integrator if not provided
        if style_integrator is None:
            self.style_integrator = StyleTransformerIntegrator(flux_transformer)
        else:
            self.style_integrator = style_integrator

    def setup_style_conditioning(self):
        """Setup style conditioning by integrating style fusion mechanisms."""
        self.style_integrator.integrate_style_fusion()

    def generate_with_style_conditioning(self,
                                       style_data: Dict[str, Any],
                                       hidden_states: torch.Tensor,
                                       encoder_hidden_states: torch.Tensor,
                                       timestep: Union[torch.Tensor, float, int],
                                       guidance: torch.Tensor = None,
                                       pooled_projections: torch.Tensor = None,
                                       return_dict: bool = True,
                                       **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate images with style conditioning.

        Args:
            style_data: Style conditioning data
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states
            timestep: Timestep for diffusion
            guidance: Guidance tensor
            pooled_projections: Pooled projections
            return_dict: Whether to return dict or tensor
            **kwargs: Additional arguments

        Returns:
            Generated output with style conditioning
        """

        # Process style data to style vectors
        if self.style_processor is not None:
            style_vectors = self.style_processor.process_style_data(style_data)
        else:
            # Assume style_data already contains style vectors
            style_vectors = style_data.get('style_vectors', None)

        # Forward pass with style conditioning
        return self.style_integrator.forward_with_style_conditioning(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            style_vectors=style_vectors,
            return_dict=return_dict,
            **kwargs
        )

    def cleanup(self):
        """Cleanup style conditioning by removing style fusion mechanisms."""
        self.style_integrator.remove_style_fusion()

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of the entire pipeline."""

        style_summary = self.style_integrator.get_integration_summary()

        return {
            'style_fusion_integration': style_summary,
            'style_processor_active': self.style_processor is not None,
            'pipeline_ready': style_summary['integration_status']
        }

    def set_style_strength(self, strength: float):
        """Set style strength for the pipeline."""
        self.style_integrator.set_style_strength(strength)

    def get_style_strength(self) -> float:
        """Get current style strength."""
        return self.style_integrator.get_style_strength()


class UnifiedBezierPipeline:
    """
    Unified pipeline that combines spatial and style conditioning for complete BezierAdapter functionality.
    """

    def __init__(self,
                 flux_transformer: FluxTransformer2DModel,
                 bezier_processor: Any = None,      # BezierParameterProcessor
                 style_processor: Any = None,       # Style feature processor
                 spatial_integrator: Any = None,    # SpatialTransformerIntegrator
                 style_integrator: StyleTransformerIntegrator = None):
        """
        Initialize the UnifiedBezierPipeline.

        Args:
            flux_transformer: FLUX transformer model
            bezier_processor: BezierParameterProcessor instance
            style_processor: Style feature processor instance
            spatial_integrator: SpatialTransformerIntegrator instance
            style_integrator: StyleTransformerIntegrator instance
        """
        self.flux_transformer = flux_transformer
        self.bezier_processor = bezier_processor
        self.style_processor = style_processor

        # Initialize integrators
        if spatial_integrator is None and bezier_processor is not None:
            from .spatial_transformer_integration import SpatialTransformerIntegrator
            self.spatial_integrator = SpatialTransformerIntegrator(flux_transformer)
        else:
            self.spatial_integrator = spatial_integrator

        if style_integrator is None:
            self.style_integrator = StyleTransformerIntegrator(flux_transformer)
        else:
            self.style_integrator = style_integrator

    def setup_unified_conditioning(self):
        """Setup both spatial and style conditioning."""
        if self.spatial_integrator is not None:
            self.spatial_integrator.integrate_spatial_attention()
        self.style_integrator.integrate_style_fusion()

    def generate_with_unified_conditioning(self,
                                         bezier_data: Dict[str, Any] = None,
                                         style_data: Dict[str, Any] = None,
                                         hidden_states: torch.Tensor = None,
                                         encoder_hidden_states: torch.Tensor = None,
                                         timestep: Union[torch.Tensor, float, int] = None,
                                         guidance: torch.Tensor = None,
                                         pooled_projections: torch.Tensor = None,
                                         return_dict: bool = True,
                                         **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate images with unified spatial and style conditioning.

        Args:
            bezier_data: Bézier curve data for spatial conditioning
            style_data: Style data for style conditioning
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states
            timestep: Timestep for diffusion
            guidance: Guidance tensor
            pooled_projections: Pooled projections
            return_dict: Whether to return dict or tensor
            **kwargs: Additional arguments

        Returns:
            Generated output with unified conditioning
        """

        # Process conditioning data
        density_map = None
        style_vectors = None

        if bezier_data is not None and self.bezier_processor is not None:
            density_map = self.bezier_processor.process_bezier_data(bezier_data)

        if style_data is not None:
            if self.style_processor is not None:
                style_vectors = self.style_processor.process_style_data(style_data)
            else:
                style_vectors = style_data.get('style_vectors', None)

        # Add conditioning to kwargs
        if density_map is not None:
            kwargs['density_map'] = density_map
        if style_vectors is not None:
            kwargs['style_vectors'] = style_vectors

        # Forward pass through the transformer
        return self.flux_transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            return_dict=return_dict,
            **kwargs
        )

    def cleanup(self):
        """Cleanup all conditioning mechanisms."""
        if self.spatial_integrator is not None:
            self.spatial_integrator.remove_spatial_attention()
        self.style_integrator.remove_style_fusion()

    def get_unified_summary(self) -> Dict[str, Any]:
        """Get summary of the unified pipeline."""

        summary = {
            'spatial_integration': None,
            'style_integration': None,
            'unified_ready': False
        }

        if self.spatial_integrator is not None:
            summary['spatial_integration'] = self.spatial_integrator.get_integration_summary()

        summary['style_integration'] = self.style_integrator.get_integration_summary()

        # Check if unified pipeline is ready
        spatial_ready = summary['spatial_integration'] is None or summary['spatial_integration']['integration_status']
        style_ready = summary['style_integration']['integration_status']
        summary['unified_ready'] = spatial_ready and style_ready

        return summary

    def set_conditioning_strength(self, style_strength: float = 1.0):
        """Set conditioning strength for the pipeline."""
        self.style_integrator.set_style_strength(style_strength)


# Utility functions for easy integration
def create_style_transformer_integrator(flux_transformer: FluxTransformer2DModel,
                                       custom_config: Dict[str, Any] = None) -> StyleTransformerIntegrator:
    """
    Create a StyleTransformerIntegrator with default or custom configuration.

    Args:
        flux_transformer: FLUX transformer model
        custom_config: Optional custom configuration

    Returns:
        Configured StyleTransformerIntegrator
    """
    return StyleTransformerIntegrator(
        flux_transformer=flux_transformer,
        style_fusion_config=custom_config
    )


def setup_bezier_style_pipeline(flux_transformer: FluxTransformer2DModel,
                               style_processor: Any = None) -> BezierStylePipeline:
    """
    Setup a complete BezierStylePipeline with style conditioning.

    Args:
        flux_transformer: FLUX transformer model
        style_processor: Style feature processor instance

    Returns:
        Configured and ready BezierStylePipeline
    """
    pipeline = BezierStylePipeline(
        flux_transformer=flux_transformer,
        style_processor=style_processor
    )

    # Setup style conditioning
    pipeline.setup_style_conditioning()

    return pipeline


def setup_unified_bezier_pipeline(flux_transformer: FluxTransformer2DModel,
                                 bezier_processor: Any = None,
                                 style_processor: Any = None) -> UnifiedBezierPipeline:
    """
    Setup a complete UnifiedBezierPipeline with both spatial and style conditioning.

    Args:
        flux_transformer: FLUX transformer model
        bezier_processor: BezierParameterProcessor instance
        style_processor: Style feature processor instance

    Returns:
        Configured and ready UnifiedBezierPipeline
    """
    pipeline = UnifiedBezierPipeline(
        flux_transformer=flux_transformer,
        bezier_processor=bezier_processor,
        style_processor=style_processor
    )

    # Setup unified conditioning
    pipeline.setup_unified_conditioning()

    return pipeline


def validate_style_integration(integrator: StyleTransformerIntegrator) -> Dict[str, Any]:
    """
    Validate style fusion integration and return diagnostic information.

    Args:
        integrator: StyleTransformerIntegrator instance

    Returns:
        Validation results and diagnostics
    """
    summary = integrator.get_integration_summary()
    breakdown = integrator.get_detailed_parameter_breakdown()

    validation_results = {
        'integration_status': summary['integration_status'],
        'parameter_count_ok': summary['total_parameters'] <= summary['target_parameters'],
        'parameter_efficiency': summary['parameter_efficiency'],
        'phase_distribution': {
            'phase2_blocks': summary['phase_breakdown']['phase2_single_stream']['num_blocks']
        },
        'detailed_breakdown': breakdown,
        'recommendations': []
    }

    # Add recommendations based on validation
    if not validation_results['parameter_count_ok']:
        validation_results['recommendations'].append(
            "Parameter count exceeds target. Consider reducing fusion_layers or attention heads."
        )

    if summary['parameter_efficiency'] < 0.8:
        validation_results['recommendations'].append(
            "Parameter efficiency is low. Consider optimizing component sizes."
        )

    if not summary['integration_status']:
        validation_results['recommendations'].append(
            "Style fusion not integrated. Call integrate_style_fusion() first."
        )

    return validation_results