import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from .spatial_attention_fuser import SpatialAttentionFuser, SpatialAttentionProcessor
from .transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock, FluxTransformer2DModel
from diffusers.models.attention_processor import FluxAttnProcessor2_0
import copy

class SpatialTransformerIntegrator(nn.Module):
    """
    SpatialTransformerIntegrator: Integrates SpatialAttentionFuser with FLUX transformer blocks.

    This module manages the integration of density-modulated attention mechanisms into
    specific transformer blocks according to the architecture mapping from TASK_01:
    - Phase 1: Double-Stream Blocks 12-18 (Enhanced Cross-Attention)
    - Phase 3: Single-Stream Blocks 20-37 (Spatial Attention Enhancement)
    """

    def __init__(self,
                 flux_transformer: FluxTransformer2DModel,
                 spatial_attention_config: Dict[str, Any] = None,
                 integration_phases: Dict[str, List[int]] = None,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the SpatialTransformerIntegrator.

        Args:
            flux_transformer: FLUX transformer model to integrate with
            spatial_attention_config: Configuration for SpatialAttentionFuser
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
            'phase1_double_stream': list(range(12, 19)),  # Double-stream blocks 12-18
            'phase3_single_stream': list(range(20, 38))   # Single-stream blocks 20-37
        }

        # Default spatial attention configuration
        self.spatial_attention_config = spatial_attention_config or {
            'hidden_dim': 3072,
            'num_heads': 24,
            'head_dim': 128,
            'density_feature_dim': 256,
            'spatial_resolution': 64,
            'fusion_layers': 3,
            'use_positional_encoding': True,
            'dropout': 0.1
        }

        # Initialize spatial attention fusers for different phases
        self._init_spatial_attention_fusers()

        # Store original attention processors for restoration
        self.original_processors = {}

        # Track integration status
        self.is_integrated = False

    def _init_spatial_attention_fusers(self):
        """Initialize spatial attention fusers for different phases."""

        # Phase 1: Double-Stream Enhanced Cross-Attention
        self.phase1_spatial_fusers = nn.ModuleDict()
        for block_idx in self.integration_phases['phase1_double_stream']:
            fuser = SpatialAttentionFuser(
                **self.spatial_attention_config,
                device=self.device,
                dtype=self.dtype
            )
            self.phase1_spatial_fusers[f'block_{block_idx}'] = fuser

        # Phase 3: Single-Stream Spatial Attention Enhancement
        self.phase3_spatial_fusers = nn.ModuleDict()
        for block_idx in self.integration_phases['phase3_single_stream']:
            fuser = SpatialAttentionFuser(
                **self.spatial_attention_config,
                device=self.device,
                dtype=self.dtype
            )
            self.phase3_spatial_fusers[f'block_{block_idx}'] = fuser

    def integrate_spatial_attention(self):
        """Integrate spatial attention mechanisms into the FLUX transformer."""

        if self.is_integrated:
            print("SpatialAttentionFuser already integrated.")
            return

        # Store original processors
        self.original_processors = copy.deepcopy(self.flux_transformer.attn_processors)

        # Get new processors with spatial attention integration
        new_processors = {}

        # Phase 1: Double-Stream Blocks Enhancement
        for block_idx in self.integration_phases['phase1_double_stream']:
            if block_idx < len(self.flux_transformer.transformer_blocks):
                block_name = f'transformer_blocks.{block_idx}.attn.processor'
                original_processor = self.flux_transformer.attn_processors.get(block_name)

                spatial_fuser = self.phase1_spatial_fusers[f'block_{block_idx}']
                spatial_processor = SpatialAttentionProcessor(
                    spatial_attention_fuser=spatial_fuser,
                    base_processor=original_processor
                )
                new_processors[block_name] = spatial_processor

        # Phase 3: Single-Stream Blocks Enhancement
        single_stream_start = len(self.flux_transformer.transformer_blocks)
        for block_idx in self.integration_phases['phase3_single_stream']:
            adjusted_idx = block_idx - single_stream_start
            if adjusted_idx < len(self.flux_transformer.single_transformer_blocks):
                block_name = f'single_transformer_blocks.{adjusted_idx}.attn.processor'
                original_processor = self.flux_transformer.attn_processors.get(block_name)

                spatial_fuser = self.phase3_spatial_fusers[f'block_{block_idx}']
                spatial_processor = SpatialAttentionProcessor(
                    spatial_attention_fuser=spatial_fuser,
                    base_processor=original_processor
                )
                new_processors[block_name] = spatial_processor

        # Update the transformer with new processors
        self.flux_transformer.set_attn_processor(new_processors)
        self.is_integrated = True

        print(f"SpatialAttentionFuser integrated into {len(new_processors)} transformer blocks.")

    def remove_spatial_attention(self):
        """Remove spatial attention mechanisms and restore original processors."""

        if not self.is_integrated:
            print("SpatialAttentionFuser not integrated.")
            return

        # Restore original processors
        self.flux_transformer.set_attn_processor(self.original_processors)
        self.is_integrated = False

        print("SpatialAttentionFuser removed. Original processors restored.")

    def forward_with_spatial_conditioning(self,
                                        hidden_states: torch.Tensor,
                                        encoder_hidden_states: torch.Tensor,
                                        timestep: Union[torch.Tensor, float, int],
                                        guidance: torch.Tensor = None,
                                        pooled_projections: torch.Tensor = None,
                                        density_map: Optional[torch.Tensor] = None,
                                        return_dict: bool = True,
                                        **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with spatial conditioning from density maps.

        Args:
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states
            timestep: Timestep for diffusion
            guidance: Guidance tensor
            pooled_projections: Pooled projections
            density_map: Density map for spatial conditioning [B, 1, H, W]
            return_dict: Whether to return dict or tensor
            **kwargs: Additional arguments

        Returns:
            Transformer output with spatial conditioning
        """

        # Store density map in kwargs for processors to access
        if density_map is not None:
            kwargs['density_map'] = density_map

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
        """Get summary of spatial attention integration."""

        total_params = 0
        phase_params = {}

        # Phase 1 parameters
        phase1_params = sum(p.numel() for fuser in self.phase1_spatial_fusers.values() for p in fuser.parameters())
        phase_params['phase1_double_stream'] = {
            'blocks': self.integration_phases['phase1_double_stream'],
            'parameters': phase1_params,
            'num_blocks': len(self.integration_phases['phase1_double_stream'])
        }
        total_params += phase1_params

        # Phase 3 parameters
        phase3_params = sum(p.numel() for fuser in self.phase3_spatial_fusers.values() for p in fuser.parameters())
        phase_params['phase3_single_stream'] = {
            'blocks': self.integration_phases['phase3_single_stream'],
            'parameters': phase3_params,
            'num_blocks': len(self.integration_phases['phase3_single_stream'])
        }
        total_params += phase3_params

        return {
            'total_parameters': total_params,
            'phase_breakdown': phase_params,
            'integration_status': self.is_integrated,
            'target_parameters': 3800000,  # 3.8M target
            'parameter_efficiency': total_params / 3800000 if total_params > 0 else 0
        }

    def get_detailed_parameter_breakdown(self) -> Dict[str, Any]:
        """Get detailed parameter breakdown for each spatial attention fuser."""

        breakdown = {
            'phase1_double_stream': {},
            'phase3_single_stream': {}
        }

        # Phase 1 breakdown
        for block_name, fuser in self.phase1_spatial_fusers.items():
            breakdown['phase1_double_stream'][block_name] = fuser.get_parameter_count()

        # Phase 3 breakdown
        for block_name, fuser in self.phase3_spatial_fusers.items():
            breakdown['phase3_single_stream'][block_name] = fuser.get_parameter_count()

        return breakdown


class BezierSpatialPipeline:
    """
    Pipeline for integrating BezierAdapter spatial conditioning with FLUX transformer.
    """

    def __init__(self,
                 flux_transformer: FluxTransformer2DModel,
                 bezier_processor: Any,  # BezierParameterProcessor
                 spatial_integrator: SpatialTransformerIntegrator = None):
        """
        Initialize the BezierSpatialPipeline.

        Args:
            flux_transformer: FLUX transformer model
            bezier_processor: BezierParameterProcessor instance
            spatial_integrator: Optional pre-configured SpatialTransformerIntegrator
        """
        self.flux_transformer = flux_transformer
        self.bezier_processor = bezier_processor

        # Initialize spatial integrator if not provided
        if spatial_integrator is None:
            self.spatial_integrator = SpatialTransformerIntegrator(flux_transformer)
        else:
            self.spatial_integrator = spatial_integrator

    def setup_spatial_conditioning(self):
        """Setup spatial conditioning by integrating spatial attention mechanisms."""
        self.spatial_integrator.integrate_spatial_attention()

    def generate_with_bezier_conditioning(self,
                                        bezier_data: Dict[str, Any],
                                        hidden_states: torch.Tensor,
                                        encoder_hidden_states: torch.Tensor,
                                        timestep: Union[torch.Tensor, float, int],
                                        guidance: torch.Tensor = None,
                                        pooled_projections: torch.Tensor = None,
                                        return_dict: bool = True,
                                        **kwargs) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate images with Bézier curve spatial conditioning.

        Args:
            bezier_data: Bézier curve data from BezierCurveExtractor
            hidden_states: Input hidden states
            encoder_hidden_states: Encoder hidden states
            timestep: Timestep for diffusion
            guidance: Guidance tensor
            pooled_projections: Pooled projections
            return_dict: Whether to return dict or tensor
            **kwargs: Additional arguments

        Returns:
            Generated output with spatial conditioning
        """

        # Process Bézier data to density map
        density_map = self.bezier_processor.process_bezier_data(bezier_data)

        # Forward pass with spatial conditioning
        return self.spatial_integrator.forward_with_spatial_conditioning(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            density_map=density_map,
            return_dict=return_dict,
            **kwargs
        )

    def cleanup(self):
        """Cleanup spatial conditioning by removing spatial attention mechanisms."""
        self.spatial_integrator.remove_spatial_attention()

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of the entire pipeline."""

        spatial_summary = self.spatial_integrator.get_integration_summary()

        return {
            'spatial_attention_integration': spatial_summary,
            'bezier_processor_active': self.bezier_processor is not None,
            'pipeline_ready': spatial_summary['integration_status']
        }


# Utility functions for easy integration
def create_spatial_attention_integrator(flux_transformer: FluxTransformer2DModel,
                                      custom_config: Dict[str, Any] = None) -> SpatialTransformerIntegrator:
    """
    Create a SpatialTransformerIntegrator with default or custom configuration.

    Args:
        flux_transformer: FLUX transformer model
        custom_config: Optional custom configuration

    Returns:
        Configured SpatialTransformerIntegrator
    """
    return SpatialTransformerIntegrator(
        flux_transformer=flux_transformer,
        spatial_attention_config=custom_config
    )


def setup_bezier_spatial_pipeline(flux_transformer: FluxTransformer2DModel,
                                 bezier_processor: Any) -> BezierSpatialPipeline:
    """
    Setup a complete BezierSpatialPipeline with spatial conditioning.

    Args:
        flux_transformer: FLUX transformer model
        bezier_processor: BezierParameterProcessor instance

    Returns:
        Configured and ready BezierSpatialPipeline
    """
    pipeline = BezierSpatialPipeline(
        flux_transformer=flux_transformer,
        bezier_processor=bezier_processor
    )

    # Setup spatial conditioning
    pipeline.setup_spatial_conditioning()

    return pipeline


def validate_spatial_integration(integrator: SpatialTransformerIntegrator) -> Dict[str, Any]:
    """
    Validate spatial attention integration and return diagnostic information.

    Args:
        integrator: SpatialTransformerIntegrator instance

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
            'phase1_blocks': summary['phase_breakdown']['phase1_double_stream']['num_blocks'],
            'phase3_blocks': summary['phase_breakdown']['phase3_single_stream']['num_blocks']
        },
        'detailed_breakdown': breakdown,
        'recommendations': []
    }

    # Add recommendations based on validation
    if not validation_results['parameter_count_ok']:
        validation_results['recommendations'].append(
            "Parameter count exceeds target. Consider reducing fusion_layers or hidden dimensions."
        )

    if summary['parameter_efficiency'] < 0.8:
        validation_results['recommendations'].append(
            "Parameter efficiency is low. Consider optimizing component sizes."
        )

    if not summary['integration_status']:
        validation_results['recommendations'].append(
            "Spatial attention not integrated. Call integrate_spatial_attention() first."
        )

    return validation_results