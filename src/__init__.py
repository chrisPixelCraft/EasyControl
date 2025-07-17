"""
EasyControl Source Package

This package provides a unified interface to all EasyControl components:
- Core: Essential EasyControl infrastructure
- BezierAdapter: Experimental Chinese calligraphy generation framework
- Examples: Demonstration scripts
- Utils: Utility functions and helpers
"""

# Core EasyControl components
from .core import (
    FluxPipeline,
    FluxTransformer2DModel,
    set_single_lora,
    set_multi_lora
)

# BezierAdapter framework (optional import)
try:
    from .bezier import (
        create_bezier_processor,
        BezierParameterProcessor,
        ConditionType,
        EnhancedMultiSingleStreamBlockLoraProcessor,
        EnhancedMultiDoubleStreamBlockLoraProcessor,
        SpatialAttentionFuser,
        StyleBezierFusionModule,
        BezierAdapterPipeline
    )
    BEZIER_ADAPTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: BezierAdapter modules not available: {e}")
    BEZIER_ADAPTER_AVAILABLE = False

# Utilities (import safely)
try:
    from .utils import *
except ImportError as e:
    print(f"Warning: Utils modules not available: {e}")
    pass

# Main exports
__all__ = [
    # Core components
    'FluxPipeline',
    'FluxTransformer2DModel',
    'set_single_lora',
    'set_multi_lora',
    
    # BezierAdapter (if available)
    'create_bezier_processor',
    'BezierParameterProcessor',
    'ConditionType',
    'EnhancedMultiSingleStreamBlockLoraProcessor',
    'EnhancedMultiDoubleStreamBlockLoraProcessor',
    'SpatialAttentionFuser',
    'StyleBezierFusionModule',
    'BezierAdapterPipeline',
    
    # Availability flag
    'BEZIER_ADAPTER_AVAILABLE'
]

# Version info
__version__ = "0.1.0"
__author__ = "EasyControl Team"
__description__ = "Efficient and flexible unified conditional diffusion transformer framework"