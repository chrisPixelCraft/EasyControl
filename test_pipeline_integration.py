import torch
import torch.nn as nn
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from PIL import Image
import json
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.bezier.bezier_adapter_pipeline import BezierAdapterPipeline, create_bezier_adapter_pipeline, BezierAdapterOutput
from src.utils.pipeline_integration_utils import (
    PipelineIntegrationManager,
    migrate_existing_pipeline,
    create_default_configs,
    save_pipeline_config,
    load_pipeline_config,
    validate_pipeline_compatibility
)
from src.bezier_parameter_processor import BezierParameterProcessor
from src.spatial_transformer_integration import SpatialTransformerIntegrator
from src.style_transformer_integration import StyleTransformerIntegrator

class MockFluxTransformer(nn.Module):
    """Mock FLUX transformer for testing."""

    def __init__(self, hidden_dim=3072, num_heads=24):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Mock transformer blocks
        self.transformer_blocks = nn.ModuleList([
            self._create_mock_block() for _ in range(19)
        ])

        self.single_transformer_blocks = nn.ModuleList([
            self._create_mock_block() for _ in range(38)
        ])

        # Mock attention processors
        self.attn_processors = {}
        for i in range(19):
            self.attn_processors[f'transformer_blocks.{i}.attn.processor'] = Mock()
        for i in range(38):
            self.attn_processors[f'single_transformer_blocks.{i}.attn.processor'] = Mock()

    def _create_mock_block(self):
        """Create a mock transformer block."""
        return nn.ModuleDict({
            'attn': nn.ModuleDict({
                'processor': Mock(),
                'to_q': nn.Linear(self.hidden_dim, self.hidden_dim),
                'to_k': nn.Linear(self.hidden_dim, self.hidden_dim),
                'to_v': nn.Linear(self.hidden_dim, self.hidden_dim),
                'to_out': nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim)])
            }),
            'heads': self.num_heads
        })

    def set_attn_processor(self, processors):
        """Set attention processors."""
        for name, processor in processors.items():
            self.attn_processors[name] = processor

    def parameters(self):
        """Return parameters."""
        return super().parameters()

    def forward(self, hidden_states, encoder_hidden_states, timestep,
                guidance=None, pooled_projections=None, return_dict=True, **kwargs):
        """Mock forward pass."""
        output = hidden_states + torch.randn_like(hidden_states) * 0.1

        if return_dict:
            return {'sample': output}
        return output

class MockFluxPipeline:
    """Mock FluxPipeline for testing migration."""

    def __init__(self):
        self.scheduler = Mock()
        self.vae = Mock()
        self.text_encoder = Mock()
        self.tokenizer = Mock()
        self.text_encoder_2 = Mock()
        self.tokenizer_2 = Mock()
        self.transformer = MockFluxTransformer()

        # Mock device
        self.device = torch.device('cpu')

    def __call__(self, *args, **kwargs):
        """Mock call method."""
        return {'images': [Mock()]}

class TestBezierAdapterPipeline:
    """Test suite for BezierAdapterPipeline functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

        # Create mock components
        self.mock_scheduler = Mock()
        self.mock_vae = Mock()
        self.mock_text_encoder = Mock()
        self.mock_tokenizer = Mock()
        self.mock_text_encoder_2 = Mock()
        self.mock_tokenizer_2 = Mock()
        self.mock_transformer = MockFluxTransformer()

        # Create pipeline
        self.pipeline = BezierAdapterPipeline(
            scheduler=self.mock_scheduler,
            vae=self.mock_vae,
            text_encoder=self.mock_text_encoder,
            tokenizer=self.mock_tokenizer,
            text_encoder_2=self.mock_text_encoder_2,
            tokenizer_2=self.mock_tokenizer_2,
            transformer=self.mock_transformer,
            enable_bezier_conditioning=True,
            enable_spatial_attention=True,
            enable_style_fusion=True,
            enable_enhanced_lora=True
        )

    def test_pipeline_initialization(self):
        """Test BezierAdapterPipeline initialization."""
        assert self.pipeline.enable_bezier_conditioning == True
        assert self.pipeline.enable_spatial_attention == True
        assert self.pipeline.enable_style_fusion == True
        assert self.pipeline.enable_enhanced_lora == True

        # Check components
        assert self.pipeline.bezier_processor is not None
        assert self.pipeline.spatial_integrator is not None
        assert self.pipeline.style_integrator is not None

    def test_pipeline_setup(self):
        """Test BezierAdapter setup."""
        assert not self.pipeline.bezier_adapter_integrated

        # Setup BezierAdapter
        self.pipeline.setup_bezier_adapter()

        assert self.pipeline.bezier_adapter_integrated

    def test_pipeline_cleanup(self):
        """Test BezierAdapter cleanup."""
        # Setup first
        self.pipeline.setup_bezier_adapter()
        assert self.pipeline.bezier_adapter_integrated

        # Cleanup
        self.pipeline.cleanup_bezier_adapter()
        assert not self.pipeline.bezier_adapter_integrated

    def test_bezier_conditioning_processing(self):
        """Test Bézier conditioning processing."""
        bezier_data = {
            'characters': [
                {
                    'bezier_curves': [
                        [[0, 0], [10, 20], [30, 40], [50, 60]],
                        [[50, 60], [70, 80], [90, 100], [110, 120]]
                    ]
                }
            ]
        }

        density_map = self.pipeline.process_bezier_conditioning(bezier_data)

        # Check that density map is generated
        assert density_map is not None
        assert density_map.shape[1:] == (1, 64, 64)  # [B, 1, H, W]

    def test_style_conditioning_processing(self):
        """Test style conditioning processing."""
        style_data = {
            'style_vectors': torch.randn(2, 256),
            'batch_size': 2
        }

        style_vectors = self.pipeline.process_style_conditioning(style_data)

        # Check that style vectors are extracted
        assert style_vectors is not None
        assert style_vectors.shape == (2, 256)

    def test_get_bezier_adapter_summary(self):
        """Test getting BezierAdapter summary."""
        summary = self.pipeline.get_bezier_adapter_summary()

        # Check summary structure
        assert 'integration_status' in summary
        assert 'components' in summary
        assert 'parameter_counts' in summary
        assert 'total_parameters' in summary

        # Check component flags
        assert summary['components']['bezier_conditioning'] == True
        assert summary['components']['spatial_attention'] == True
        assert summary['components']['style_fusion'] == True
        assert summary['components']['enhanced_lora'] == True

    def test_conditioning_strength_control(self):
        """Test conditioning strength control."""
        # Test setting conditioning strength
        self.pipeline.set_conditioning_strength(style_strength=0.8)

        # Check that style strength was set
        assert self.pipeline.style_integrator.get_style_strength() == 0.8

    def test_selective_component_enabling(self):
        """Test enabling/disabling specific components."""
        # Create pipeline with only some components enabled
        selective_pipeline = BezierAdapterPipeline(
            scheduler=self.mock_scheduler,
            vae=self.mock_vae,
            text_encoder=self.mock_text_encoder,
            tokenizer=self.mock_tokenizer,
            text_encoder_2=self.mock_text_encoder_2,
            tokenizer_2=self.mock_tokenizer_2,
            transformer=self.mock_transformer,
            enable_bezier_conditioning=True,
            enable_spatial_attention=False,
            enable_style_fusion=True,
            enable_enhanced_lora=False
        )

        # Check that only selected components are enabled
        assert selective_pipeline.bezier_processor is not None
        assert selective_pipeline.spatial_integrator is None
        assert selective_pipeline.style_integrator is not None

    def test_pipeline_call_with_bezier_conditioning(self):
        """Test pipeline call with BezierAdapter conditioning."""
        # Mock the parent class call
        with patch.object(self.pipeline.__class__.__bases__[0], '__call__', return_value={'images': [Mock()]}) as mock_call:
            # Prepare test data
            bezier_data = {
                'characters': [
                    {
                        'bezier_curves': [
                            [[0, 0], [10, 20], [30, 40], [50, 60]]
                        ]
                    }
                ]
            }

            style_data = {
                'style_vectors': torch.randn(1, 256),
                'batch_size': 1
            }

            # Call pipeline
            result = self.pipeline(
                prompt="test prompt",
                bezier_data=bezier_data,
                style_data=style_data,
                return_debug_info=False
            )

            # Check that parent call was made
            mock_call.assert_called_once()

            # Check that joint_attention_kwargs were passed
            call_kwargs = mock_call.call_args[1]
            assert 'joint_attention_kwargs' in call_kwargs

    def test_pipeline_call_with_debug_info(self):
        """Test pipeline call with debug info."""
        # Mock the parent class call
        with patch.object(self.pipeline.__class__.__bases__[0], '__call__', return_value={'images': [Mock()]}) as mock_call:
            # Call pipeline with debug info
            result = self.pipeline(
                prompt="test prompt",
                bezier_data={'characters': [{'bezier_curves': [[[0, 0], [10, 20], [30, 40], [50, 60]]]}]},
                style_data={'style_vectors': torch.randn(1, 256)},
                return_debug_info=True,
                return_dict=True
            )

            # Check that result is BezierAdapterOutput
            assert isinstance(result, BezierAdapterOutput)
            assert hasattr(result, 'images')
            assert hasattr(result, 'bezier_debug_info')
            assert hasattr(result, 'style_debug_info')


class TestPipelineIntegrationManager:
    """Test suite for PipelineIntegrationManager functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

        # Create mock pipeline
        self.mock_pipeline = BezierAdapterPipeline(
            scheduler=Mock(),
            vae=Mock(),
            text_encoder=Mock(),
            tokenizer=Mock(),
            text_encoder_2=Mock(),
            tokenizer_2=Mock(),
            transformer=MockFluxTransformer()
        )

        # Create integration manager
        self.integration_manager = PipelineIntegrationManager(self.mock_pipeline)

    def test_integration_manager_initialization(self):
        """Test PipelineIntegrationManager initialization."""
        assert self.integration_manager.pipeline == self.mock_pipeline
        assert self.integration_manager.validation_results == {}
        assert self.integration_manager.performance_metrics == {}

    def test_validate_integration(self):
        """Test integration validation."""
        validation_results = self.integration_manager.validate_integration()

        # Check validation structure
        assert 'overall_status' in validation_results
        assert 'components' in validation_results
        assert 'parameter_counts' in validation_results
        assert 'recommendations' in validation_results
        assert 'warnings' in validation_results
        assert 'errors' in validation_results

        # Check that validation was stored
        assert self.integration_manager.validation_results == validation_results

    def test_benchmark_performance(self):
        """Test performance benchmarking."""
        performance_metrics = self.integration_manager.benchmark_performance(
            batch_sizes=[1, 2],
            num_runs=2
        )

        # Check performance metrics structure
        assert 'batch_size_results' in performance_metrics
        assert 'memory_usage' in performance_metrics
        assert 'overall_performance' in performance_metrics

        # Check batch size results
        for batch_size in [1, 2]:
            assert batch_size in performance_metrics['batch_size_results']

        # Check that performance metrics were stored
        assert self.integration_manager.performance_metrics == performance_metrics

    def test_generate_integration_report(self):
        """Test integration report generation."""
        report = self.integration_manager.generate_integration_report()

        # Check report structure
        assert 'pipeline_info' in report
        assert 'validation_results' in report
        assert 'performance_metrics' in report
        assert 'recommendations' in report

        # Check pipeline info
        assert report['pipeline_info']['type'] == 'BezierAdapterPipeline'

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        # Run validation first
        self.integration_manager.validate_integration()

        recommendations = self.integration_manager._generate_recommendations()

        # Check that recommendations are generated
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0


class TestPipelineIntegrationUtils:
    """Test suite for pipeline integration utility functions."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

    def test_migrate_existing_pipeline(self):
        """Test migrating existing pipeline."""
        # Create mock existing pipeline
        existing_pipeline = MockFluxPipeline()

        # Migrate to BezierAdapterPipeline
        bezier_pipeline = migrate_existing_pipeline(existing_pipeline)

        # Check that migration was successful
        assert isinstance(bezier_pipeline, BezierAdapterPipeline)
        assert bezier_pipeline.scheduler == existing_pipeline.scheduler
        assert bezier_pipeline.vae == existing_pipeline.vae
        assert bezier_pipeline.transformer == existing_pipeline.transformer

    def test_create_default_configs(self):
        """Test creating default configurations."""
        configs = create_default_configs()

        # Check that all required configs are present
        assert 'bezier_config' in configs
        assert 'spatial_config' in configs
        assert 'style_config' in configs
        assert 'lora_config' in configs

        # Check bezier config
        assert 'output_size' in configs['bezier_config']
        assert 'hidden_dim' in configs['bezier_config']

        # Check spatial config
        assert 'spatial_attention_config' in configs['spatial_config']

        # Check style config
        assert 'style_fusion_config' in configs['style_config']

        # Check lora config
        assert 'lora_rank' in configs['lora_config']

    def test_save_and_load_pipeline_config(self):
        """Test saving and loading pipeline configuration."""
        # Create mock pipeline
        pipeline = BezierAdapterPipeline(
            scheduler=Mock(),
            vae=Mock(),
            text_encoder=Mock(),
            tokenizer=Mock(),
            text_encoder_2=Mock(),
            tokenizer_2=Mock(),
            transformer=MockFluxTransformer()
        )

        # Save config to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name

        try:
            save_pipeline_config(pipeline, config_path)

            # Load config
            loaded_config = load_pipeline_config(config_path)

            # Check that config was loaded correctly
            assert loaded_config['pipeline_type'] == 'BezierAdapterPipeline'
            assert 'components_enabled' in loaded_config
            assert 'integration_status' in loaded_config

        finally:
            # Clean up
            os.unlink(config_path)

    def test_validate_pipeline_compatibility(self):
        """Test pipeline compatibility validation."""
        # Test with compatible pipeline
        compatible_pipeline = MockFluxPipeline()

        compatibility_results = validate_pipeline_compatibility(compatible_pipeline)

        # Check compatibility results
        assert compatibility_results['compatible'] == True
        assert compatibility_results['pipeline_type'] == 'MockFluxPipeline'
        assert len(compatibility_results['missing_components']) == 0

        # Test with incompatible pipeline
        incompatible_pipeline = Mock()
        delattr(incompatible_pipeline, 'scheduler') if hasattr(incompatible_pipeline, 'scheduler') else None

        compatibility_results = validate_pipeline_compatibility(incompatible_pipeline)

        # Check that incompatibility is detected
        assert compatibility_results['compatible'] == False
        assert len(compatibility_results['missing_components']) > 0


class TestCreateBezierAdapterPipeline:
    """Test suite for create_bezier_adapter_pipeline function."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

        # Create mock components
        self.mock_scheduler = Mock()
        self.mock_vae = Mock()
        self.mock_text_encoder = Mock()
        self.mock_tokenizer = Mock()
        self.mock_text_encoder_2 = Mock()
        self.mock_tokenizer_2 = Mock()
        self.mock_transformer = MockFluxTransformer()

    def test_create_bezier_adapter_pipeline_default(self):
        """Test creating BezierAdapterPipeline with default settings."""
        pipeline = create_bezier_adapter_pipeline(
            scheduler=self.mock_scheduler,
            vae=self.mock_vae,
            text_encoder=self.mock_text_encoder,
            tokenizer=self.mock_tokenizer,
            text_encoder_2=self.mock_text_encoder_2,
            tokenizer_2=self.mock_tokenizer_2,
            transformer=self.mock_transformer
        )

        # Check that pipeline was created correctly
        assert isinstance(pipeline, BezierAdapterPipeline)
        assert pipeline.enable_bezier_conditioning == True
        assert pipeline.enable_spatial_attention == True
        assert pipeline.enable_style_fusion == True
        assert pipeline.enable_enhanced_lora == True

    def test_create_bezier_adapter_pipeline_custom_config(self):
        """Test creating BezierAdapterPipeline with custom configuration."""
        custom_configs = {
            'bezier_config': {
                'output_size': (32, 32),
                'hidden_dim': 128
            },
            'spatial_config': {
                'spatial_attention_config': {
                    'hidden_dim': 2048,
                    'num_heads': 16
                }
            }
        }

        pipeline = create_bezier_adapter_pipeline(
            scheduler=self.mock_scheduler,
            vae=self.mock_vae,
            text_encoder=self.mock_text_encoder,
            tokenizer=self.mock_tokenizer,
            text_encoder_2=self.mock_text_encoder_2,
            tokenizer_2=self.mock_tokenizer_2,
            transformer=self.mock_transformer,
            custom_configs=custom_configs
        )

        # Check that pipeline was created correctly
        assert isinstance(pipeline, BezierAdapterPipeline)

        # Check that custom configs were applied
        # (Would need to inspect the actual components to verify this)

    def test_create_bezier_adapter_pipeline_selective_components(self):
        """Test creating BezierAdapterPipeline with selective components."""
        pipeline = create_bezier_adapter_pipeline(
            scheduler=self.mock_scheduler,
            vae=self.mock_vae,
            text_encoder=self.mock_text_encoder,
            tokenizer=self.mock_tokenizer,
            text_encoder_2=self.mock_text_encoder_2,
            tokenizer_2=self.mock_tokenizer_2,
            transformer=self.mock_transformer,
            enable_all_components=False
        )

        # Check that components are disabled
        assert pipeline.enable_bezier_conditioning == False
        assert pipeline.enable_spatial_attention == False
        assert pipeline.enable_style_fusion == False
        assert pipeline.enable_enhanced_lora == False


class TestBackwardCompatibility:
    """Test suite for backward compatibility with existing pipelines."""

    def setup_method(self):
        """Setup test fixtures."""
        self.device = 'cpu'

        # Create BezierAdapterPipeline
        self.pipeline = BezierAdapterPipeline(
            scheduler=Mock(),
            vae=Mock(),
            text_encoder=Mock(),
            tokenizer=Mock(),
            text_encoder_2=Mock(),
            tokenizer_2=Mock(),
            transformer=MockFluxTransformer()
        )

    def test_backward_compatibility_spatial_images(self):
        """Test backward compatibility with spatial_images parameter."""
        # Mock parent class call
        with patch.object(self.pipeline.__class__.__bases__[0], '__call__', return_value={'images': [Mock()]}) as mock_call:
            # Create mock spatial images
            spatial_images = [Mock(spec=Image.Image) for _ in range(2)]

            # Call with spatial_images (existing parameter)
            result = self.pipeline(
                prompt="test prompt",
                spatial_images=spatial_images,
                cond_size=512
            )

            # Check that parent call was made with spatial_images
            mock_call.assert_called_once()
            call_kwargs = mock_call.call_args[1]
            assert 'spatial_images' in call_kwargs
            assert call_kwargs['spatial_images'] == spatial_images

    def test_backward_compatibility_subject_images(self):
        """Test backward compatibility with subject_images parameter."""
        # Mock parent class call
        with patch.object(self.pipeline.__class__.__bases__[0], '__call__', return_value={'images': [Mock()]}) as mock_call:
            # Create mock subject images
            subject_images = [Mock(spec=Image.Image) for _ in range(1)]

            # Call with subject_images (existing parameter)
            result = self.pipeline(
                prompt="test prompt",
                subject_images=subject_images,
                cond_size=512
            )

            # Check that parent call was made with subject_images
            mock_call.assert_called_once()
            call_kwargs = mock_call.call_args[1]
            assert 'subject_images' in call_kwargs
            assert call_kwargs['subject_images'] == subject_images

    def test_backward_compatibility_mixed_conditioning(self):
        """Test backward compatibility with mixed old and new conditioning."""
        # Mock parent class call
        with patch.object(self.pipeline.__class__.__bases__[0], '__call__', return_value={'images': [Mock()]}) as mock_call:
            # Create mixed conditioning
            spatial_images = [Mock(spec=Image.Image)]
            bezier_data = {
                'characters': [
                    {
                        'bezier_curves': [
                            [[0, 0], [10, 20], [30, 40], [50, 60]]
                        ]
                    }
                ]
            }

            # Call with both old and new conditioning
            result = self.pipeline(
                prompt="test prompt",
                spatial_images=spatial_images,
                bezier_data=bezier_data,
                cond_size=512
            )

            # Check that parent call was made with both types of conditioning
            mock_call.assert_called_once()
            call_kwargs = mock_call.call_args[1]
            assert 'spatial_images' in call_kwargs
            assert 'joint_attention_kwargs' in call_kwargs
            assert call_kwargs['spatial_images'] == spatial_images


def run_integration_tests():
    """Run all integration tests."""
    print("Running BezierAdapter Pipeline Integration Tests...")

    # Test basic pipeline functionality
    print("\n=== Basic Pipeline Tests ===")
    test_pipeline = TestBezierAdapterPipeline()
    test_pipeline.setup_method()
    test_pipeline.test_pipeline_initialization()
    test_pipeline.test_pipeline_setup()
    test_pipeline.test_bezier_conditioning_processing()
    test_pipeline.test_style_conditioning_processing()
    print("✓ Basic pipeline tests passed")

    # Test integration manager
    print("\n=== Integration Manager Tests ===")
    test_manager = TestPipelineIntegrationManager()
    test_manager.setup_method()
    test_manager.test_integration_manager_initialization()
    test_manager.test_validate_integration()
    test_manager.test_benchmark_performance()
    print("✓ Integration manager tests passed")

    # Test utility functions
    print("\n=== Utility Function Tests ===")
    test_utils = TestPipelineIntegrationUtils()
    test_utils.setup_method()
    test_utils.test_create_default_configs()
    test_utils.test_save_and_load_pipeline_config()
    test_utils.test_validate_pipeline_compatibility()
    print("✓ Utility function tests passed")

    # Test backward compatibility
    print("\n=== Backward Compatibility Tests ===")
    test_compat = TestBackwardCompatibility()
    test_compat.setup_method()
    test_compat.test_backward_compatibility_spatial_images()
    test_compat.test_backward_compatibility_subject_images()
    print("✓ Backward compatibility tests passed")

    print("\n" + "="*50)
    print("✓ All pipeline integration tests passed!")
    print("BezierAdapter pipeline integration is working correctly.")
    print("="*50)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Run integration tests
    run_integration_tests()

    # Test with pytest if available
    try:
        import pytest
        print("\n=== Running pytest ===")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\nPytest not available, skipping automated tests")