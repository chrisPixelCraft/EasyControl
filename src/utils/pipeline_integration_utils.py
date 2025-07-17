"""
Pipeline Integration Utilities

This module provides utility functions for integrating BezierAdapter components
with existing FLUX pipelines, including validation, configuration helpers,
and migration utilities.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
import json
import os
from pathlib import Path

from ..bezier.bezier_adapter_pipeline import BezierAdapterPipeline, create_bezier_adapter_pipeline
from ..bezier.bezier_parameter_processor import BezierParameterProcessor
from .spatial_transformer_integration import SpatialTransformerIntegrator, validate_spatial_integration
from .style_transformer_integration import StyleTransformerIntegrator, validate_style_integration

class PipelineIntegrationManager:
    """
    Manager class for handling pipeline integration operations.

    This class provides utilities for:
    - Validating pipeline configurations
    - Migrating existing pipelines to BezierAdapter
    - Managing component integration
    - Performance monitoring
    """

    def __init__(self, pipeline: BezierAdapterPipeline):
        """
        Initialize the integration manager.

        Args:
            pipeline: BezierAdapterPipeline instance
        """
        self.pipeline = pipeline
        self.validation_results = {}
        self.performance_metrics = {}

    def validate_integration(self, detailed: bool = True) -> Dict[str, Any]:
        """
        Validate the entire BezierAdapter integration.

        Args:
            detailed: Whether to include detailed validation results

        Returns:
            Validation results dictionary
        """

        validation_results = {
            'overall_status': 'pending',
            'components': {},
            'parameter_counts': {},
            'recommendations': [],
            'warnings': [],
            'errors': []
        }

        # 1. Validate BezierParameterProcessor
        if self.pipeline.bezier_processor is not None:
            try:
                # Test basic functionality
                test_bezier_data = {
                    'characters': [
                        {
                            'bezier_curves': [
                                [[0, 0], [10, 20], [30, 40], [50, 60]],
                                [[50, 60], [70, 80], [90, 100], [110, 120]]
                            ]
                        }
                    ]
                }

                density_map = self.pipeline.bezier_processor.process_bezier_data(test_bezier_data)

                validation_results['components']['bezier_processor'] = {
                    'status': 'valid',
                    'output_shape': density_map.shape,
                    'parameter_count': sum(p.numel() for p in self.pipeline.bezier_processor.parameters())
                }

            except Exception as e:
                validation_results['components']['bezier_processor'] = {
                    'status': 'error',
                    'error': str(e)
                }
                validation_results['errors'].append(f"BezierParameterProcessor error: {e}")

        # 2. Validate SpatialTransformerIntegrator
        if self.pipeline.spatial_integrator is not None:
            try:
                spatial_validation = validate_spatial_integration(self.pipeline.spatial_integrator)
                validation_results['components']['spatial_integrator'] = spatial_validation

                if not spatial_validation['parameter_count_ok']:
                    validation_results['warnings'].append("SpatialAttentionFuser parameter count exceeds target")

            except Exception as e:
                validation_results['components']['spatial_integrator'] = {
                    'status': 'error',
                    'error': str(e)
                }
                validation_results['errors'].append(f"SpatialTransformerIntegrator error: {e}")

        # 3. Validate StyleTransformerIntegrator
        if self.pipeline.style_integrator is not None:
            try:
                style_validation = validate_style_integration(self.pipeline.style_integrator)
                validation_results['components']['style_integrator'] = style_validation

                if not style_validation['parameter_count_ok']:
                    validation_results['warnings'].append("StyleBezierFusionModule parameter count exceeds target")

            except Exception as e:
                validation_results['components']['style_integrator'] = {
                    'status': 'error',
                    'error': str(e)
                }
                validation_results['errors'].append(f"StyleTransformerIntegrator error: {e}")

        # 4. Validate overall integration
        if self.pipeline.bezier_adapter_integrated:
            validation_results['components']['integration_status'] = 'integrated'
        else:
            validation_results['components']['integration_status'] = 'not_integrated'
            validation_results['warnings'].append("BezierAdapter components not integrated")

        # 5. Check parameter counts
        summary = self.pipeline.get_bezier_adapter_summary()
        validation_results['parameter_counts'] = summary['parameter_counts']
        validation_results['parameter_counts']['total'] = summary['total_parameters']

        # Target: ~18.5M parameters
        target_params = 18500000
        if summary['total_parameters'] > target_params:
            validation_results['warnings'].append(f"Total parameters ({summary['total_parameters']:,}) exceed target ({target_params:,})")

        # 6. Generate recommendations
        if len(validation_results['errors']) == 0:
            if len(validation_results['warnings']) == 0:
                validation_results['overall_status'] = 'valid'
                validation_results['recommendations'].append("Integration is optimal")
            else:
                validation_results['overall_status'] = 'valid_with_warnings'
                validation_results['recommendations'].append("Integration is functional but has warnings")
        else:
            validation_results['overall_status'] = 'invalid'
            validation_results['recommendations'].append("Fix errors before using the pipeline")

        self.validation_results = validation_results
        return validation_results

    def benchmark_performance(self,
                            batch_sizes: List[int] = [1, 2, 4],
                            num_runs: int = 3) -> Dict[str, Any]:
        """
        Benchmark performance of the BezierAdapter pipeline.

        Args:
            batch_sizes: List of batch sizes to test
            num_runs: Number of runs per batch size

        Returns:
            Performance metrics
        """

        import time

        performance_metrics = {
            'batch_size_results': {},
            'component_breakdown': {},
            'memory_usage': {},
            'overall_performance': {}
        }

        # Test different batch sizes
        for batch_size in batch_sizes:
            batch_results = []

            for run in range(num_runs):
                # Create test data
                test_bezier_data = {
                    'characters': [
                        {
                            'bezier_curves': [
                                [[0, 0], [10, 20], [30, 40], [50, 60]],
                                [[50, 60], [70, 80], [90, 100], [110, 120]]
                            ]
                        }
                    ]
                }

                test_style_data = {
                    'style_vectors': torch.randn(batch_size, 256, device=self.pipeline.device),
                    'batch_size': batch_size
                }

                # Measure processing time
                start_time = time.time()

                try:
                    # Process conditioning
                    density_map = self.pipeline.process_bezier_conditioning(test_bezier_data)
                    style_vectors = self.pipeline.process_style_conditioning(test_style_data)

                    processing_time = time.time() - start_time

                    batch_results.append({
                        'processing_time': processing_time,
                        'density_map_shape': density_map.shape if density_map is not None else None,
                        'style_vectors_shape': style_vectors.shape if style_vectors is not None else None,
                        'success': True
                    })

                except Exception as e:
                    batch_results.append({
                        'processing_time': None,
                        'error': str(e),
                        'success': False
                    })

            # Calculate statistics
            successful_runs = [r for r in batch_results if r['success']]

            if successful_runs:
                avg_time = sum(r['processing_time'] for r in successful_runs) / len(successful_runs)
                min_time = min(r['processing_time'] for r in successful_runs)
                max_time = max(r['processing_time'] for r in successful_runs)

                performance_metrics['batch_size_results'][batch_size] = {
                    'avg_processing_time': avg_time,
                    'min_processing_time': min_time,
                    'max_processing_time': max_time,
                    'success_rate': len(successful_runs) / len(batch_results),
                    'total_runs': len(batch_results)
                }
            else:
                performance_metrics['batch_size_results'][batch_size] = {
                    'avg_processing_time': None,
                    'success_rate': 0,
                    'total_runs': len(batch_results),
                    'errors': [r['error'] for r in batch_results if not r['success']]
                }

        # Memory usage
        if torch.cuda.is_available():
            performance_metrics['memory_usage'] = {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'device': 'cuda'
            }
        else:
            performance_metrics['memory_usage'] = {
                'device': 'cpu'
            }

        # Overall performance assessment
        successful_batches = [k for k, v in performance_metrics['batch_size_results'].items() if v['success_rate'] > 0.5]

        if successful_batches:
            avg_times = [performance_metrics['batch_size_results'][b]['avg_processing_time'] for b in successful_batches]
            performance_metrics['overall_performance'] = {
                'status': 'good' if all(t < 1.0 for t in avg_times) else 'acceptable',
                'avg_processing_time': sum(avg_times) / len(avg_times),
                'throughput_estimate': 1.0 / (sum(avg_times) / len(avg_times))
            }
        else:
            performance_metrics['overall_performance'] = {
                'status': 'poor',
                'avg_processing_time': None,
                'throughput_estimate': 0
            }

        self.performance_metrics = performance_metrics
        return performance_metrics

    def generate_integration_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive integration report.

        Args:
            output_path: Optional path to save the report

        Returns:
            Integration report
        """

        # Run validation if not already done
        if not self.validation_results:
            self.validate_integration()

        # Run performance benchmark if not already done
        if not self.performance_metrics:
            self.benchmark_performance()

        # Generate report
        report = {
            'pipeline_info': {
                'type': 'BezierAdapterPipeline',
                'components_enabled': self.pipeline.get_bezier_adapter_summary()['components'],
                'integration_status': self.pipeline.bezier_adapter_integrated,
                'device': str(self.pipeline.device)
            },
            'validation_results': self.validation_results,
            'performance_metrics': self.performance_metrics,
            'recommendations': self._generate_recommendations(),
            'timestamp': torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else 'unavailable'
        }

        # Save report if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"Integration report saved to: {output_path}")

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation and performance results."""

        recommendations = []

        # Based on validation results
        if self.validation_results:
            if self.validation_results['overall_status'] == 'valid':
                recommendations.append("✓ Integration is working correctly")
            elif self.validation_results['overall_status'] == 'valid_with_warnings':
                recommendations.append("⚠ Integration is functional but consider addressing warnings")
            else:
                recommendations.append("❌ Fix validation errors before using the pipeline")

        # Based on performance results
        if self.performance_metrics:
            overall_perf = self.performance_metrics.get('overall_performance', {})
            if overall_perf.get('status') == 'good':
                recommendations.append("✓ Performance is good")
            elif overall_perf.get('status') == 'acceptable':
                recommendations.append("⚠ Performance is acceptable but could be improved")
            else:
                recommendations.append("❌ Performance issues detected")

        return recommendations


def migrate_existing_pipeline(
    existing_pipeline: Any,
    enable_bezier_conditioning: bool = True,
    enable_spatial_attention: bool = True,
    enable_style_fusion: bool = True,
    enable_enhanced_lora: bool = True
) -> BezierAdapterPipeline:
    """
    Migrate an existing FluxPipeline to BezierAdapterPipeline.

    Args:
        existing_pipeline: Existing FluxPipeline instance
        enable_bezier_conditioning: Whether to enable Bézier conditioning
        enable_spatial_attention: Whether to enable spatial attention
        enable_style_fusion: Whether to enable style fusion
        enable_enhanced_lora: Whether to enable enhanced LoRA

    Returns:
        Migrated BezierAdapterPipeline
    """

    # Extract components from existing pipeline
    bezier_pipeline = BezierAdapterPipeline(
        scheduler=existing_pipeline.scheduler,
        vae=existing_pipeline.vae,
        text_encoder=existing_pipeline.text_encoder,
        tokenizer=existing_pipeline.tokenizer,
        text_encoder_2=existing_pipeline.text_encoder_2,
        tokenizer_2=existing_pipeline.tokenizer_2,
        transformer=existing_pipeline.transformer,
        enable_bezier_conditioning=enable_bezier_conditioning,
        enable_spatial_attention=enable_spatial_attention,
        enable_style_fusion=enable_style_fusion,
        enable_enhanced_lora=enable_enhanced_lora
    )

    print("Pipeline migration completed successfully")
    return bezier_pipeline


def create_default_configs() -> Dict[str, Dict[str, Any]]:
    """
    Create default configurations for all BezierAdapter components.

    Returns:
        Dictionary of default configurations
    """

    configs = {
        'bezier_config': {
            'output_size': (64, 64),
            'hidden_dim': 256,
            'num_gaussian_components': 8,
            'curve_resolution': 100,
            'density_sigma': 2.0,
            'learnable_kde': True
        },
        'spatial_config': {
            'spatial_attention_config': {
                'hidden_dim': 3072,
                'num_heads': 24,
                'head_dim': 128,
                'density_feature_dim': 256,
                'spatial_resolution': 64,
                'fusion_layers': 3,
                'use_positional_encoding': True,
                'dropout': 0.1
            }
        },
        'style_config': {
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
            }
        },
        'lora_config': {
            'lora_rank': 64,
            'lora_alpha': 64,
            'lora_dropout': 0.1,
            'bezier_condition_types': ['style', 'text', 'density']
        }
    }

    return configs


def save_pipeline_config(pipeline: BezierAdapterPipeline, config_path: str):
    """
    Save pipeline configuration to file.

    Args:
        pipeline: BezierAdapterPipeline instance
        config_path: Path to save configuration
    """

    config = {
        'pipeline_type': 'BezierAdapterPipeline',
        'components_enabled': {
            'bezier_conditioning': pipeline.enable_bezier_conditioning,
            'spatial_attention': pipeline.enable_spatial_attention,
            'style_fusion': pipeline.enable_style_fusion,
            'enhanced_lora': pipeline.enable_enhanced_lora
        },
        'integration_status': pipeline.bezier_adapter_integrated,
        'parameter_summary': pipeline.get_bezier_adapter_summary(),
        'device': str(pipeline.device)
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"Pipeline configuration saved to: {config_path}")


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """
    Load pipeline configuration from file.

    Args:
        config_path: Path to configuration file

    Returns:
        Pipeline configuration dictionary
    """

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def validate_pipeline_compatibility(pipeline: Any) -> Dict[str, Any]:
    """
    Validate if a pipeline is compatible with BezierAdapter integration.

    Args:
        pipeline: Pipeline to validate

    Returns:
        Compatibility validation results
    """

    compatibility_results = {
        'compatible': False,
        'pipeline_type': type(pipeline).__name__,
        'required_components': [],
        'missing_components': [],
        'warnings': []
    }

    # Check required components
    required_components = ['scheduler', 'vae', 'text_encoder', 'tokenizer', 'text_encoder_2', 'tokenizer_2', 'transformer']

    for component in required_components:
        if hasattr(pipeline, component):
            compatibility_results['required_components'].append(component)
        else:
            compatibility_results['missing_components'].append(component)

    # Check if all required components are present
    if len(compatibility_results['missing_components']) == 0:
        compatibility_results['compatible'] = True
    else:
        compatibility_results['warnings'].append(f"Missing components: {compatibility_results['missing_components']}")

    # Check transformer type
    if hasattr(pipeline, 'transformer'):
        if not isinstance(pipeline.transformer, FluxTransformer2DModel):
            compatibility_results['warnings'].append("Transformer is not FluxTransformer2DModel")

    return compatibility_results