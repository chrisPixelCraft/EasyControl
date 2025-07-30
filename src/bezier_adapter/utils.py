"""
BezierAdapter Utilities
======================

Core data structures and mathematical functions for Bézier curve processing.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from scipy.stats import gaussian_kde


@dataclass
class BezierCurve:
    """
    Represents a single Bézier curve with control points and metadata.
    
    Args:
        control_points: [4, 2] for cubic Bézier in normalized coords [-1, 1]
        curve_id: Optional identifier for the curve
        weight: Importance weight for the curve (default: 1.0)
    """
    control_points: torch.Tensor  
    curve_id: Optional[str] = None
    weight: float = 1.0
    
    def __post_init__(self):
        """Validate control points dimensions and normalize if needed."""
        if self.control_points.shape != (4, 2):
            raise ValueError(f"Expected control_points shape (4, 2), got {self.control_points.shape}")
        
        # Ensure tensor is float type
        if self.control_points.dtype != torch.float32:
            self.control_points = self.control_points.float()


@dataclass
class BezierConfig:
    """
    Configuration for Bézier processing parameters.
    
    Args:
        max_curves: Maximum number of curves per batch
        max_points_per_curve: Maximum control points per curve (4 for cubic)
        output_resolution: Target output resolution (H, W)
        kde_bandwidth: Gaussian KDE bandwidth for density calculation
        density_threshold: Threshold for density map values
    """
    max_curves: int = 16
    max_points_per_curve: int = 8
    output_resolution: Tuple[int, int] = (256, 256)
    kde_bandwidth: float = 0.1
    density_threshold: float = 0.5


@dataclass
class DensityMap:
    """
    Processed density map with metadata.
    
    Args:
        density: [B, 1, H, W] density weights [0, 1]
        field_map: [B, 2, H, W] control field vectors [-1, 1]
        source_curves: List of source Bézier curves
        resolution: Output resolution (H, W)
    """
    density: torch.Tensor
    field_map: torch.Tensor        
    source_curves: List[BezierCurve]
    resolution: Tuple[int, int]
    
    def __post_init__(self):
        """Validate tensor dimensions and ranges."""
        B, C, H, W = self.density.shape
        if C != 1:
            raise ValueError(f"Expected density channel dimension 1, got {C}")
        
        if self.field_map.shape != (B, 2, H, W):
            raise ValueError(f"Expected field_map shape ({B}, 2, {H}, {W}), got {self.field_map.shape}")


@dataclass 
class LoRAConfig:
    """
    LoRA configuration for parameter-efficient adaptation.
    
    Args:
        rank: Rank of LoRA decomposition
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA layers
        target_modules: List of module names to apply LoRA
    """
    rank: int = 64
    alpha: float = 64.0
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"])


def cubic_bezier_point(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Evaluate cubic Bézier curve at parameter t using De Casteljau's algorithm.
    
    Args:
        control_points: [4, 2] control points P0, P1, P2, P3
        t: [N] parameter values in [0, 1]
        
    Returns:
        points: [N, 2] evaluated points on curve
    """
    # Ensure t is proper shape
    if t.dim() == 0:
        t = t.unsqueeze(0)
    
    # De Casteljau's algorithm for numerical stability
    P0, P1, P2, P3 = control_points[0], control_points[1], control_points[2], control_points[3]
    
    # First level interpolation
    Q0 = (1 - t.unsqueeze(-1)) * P0 + t.unsqueeze(-1) * P1  # [N, 2]
    Q1 = (1 - t.unsqueeze(-1)) * P1 + t.unsqueeze(-1) * P2
    Q2 = (1 - t.unsqueeze(-1)) * P2 + t.unsqueeze(-1) * P3
    
    # Second level interpolation
    R0 = (1 - t.unsqueeze(-1)) * Q0 + t.unsqueeze(-1) * Q1
    R1 = (1 - t.unsqueeze(-1)) * Q1 + t.unsqueeze(-1) * Q2
    
    # Final interpolation
    curve_point = (1 - t.unsqueeze(-1)) * R0 + t.unsqueeze(-1) * R1
    
    return curve_point


def evaluate_curve(control_points: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
    """
    Evaluate Bézier curve at evenly spaced parameter values.
    
    Args:
        control_points: [4, 2] cubic Bézier control points
        num_samples: Number of points to sample along curve
        
    Returns:
        curve_points: [num_samples, 2] sampled points
    """
    device = control_points.device
    t_values = torch.linspace(0, 1, num_samples, device=device)
    return cubic_bezier_point(control_points, t_values)


def normalize_coordinates(coordinates: torch.Tensor, 
                        source_range: Tuple[float, float] = (0, 256),
                        target_range: Tuple[float, float] = (-1, 1)) -> torch.Tensor:
    """
    Normalize coordinates from source range to target range.
    
    Args:
        coordinates: [..., 2] coordinate tensor
        source_range: (min, max) of source coordinate system  
        target_range: (min, max) of target coordinate system
        
    Returns:
        normalized: [..., 2] normalized coordinates
    """
    source_min, source_max = source_range
    target_min, target_max = target_range
    
    # Normalize to [0, 1]
    normalized = (coordinates - source_min) / (source_max - source_min)
    
    # Scale to target range
    normalized = normalized * (target_max - target_min) + target_min
    
    return normalized


def gaussian_kde_density(points: torch.Tensor, 
                        grid_points: torch.Tensor,
                        bandwidth: float = 0.1) -> torch.Tensor:
    """
    Compute Gaussian KDE density estimation at grid points.
    
    Args:
        points: [N, 2] input points
        grid_points: [H*W, 2] grid points for evaluation
        bandwidth: KDE bandwidth parameter
        
    Returns:
        density: [H*W] density values at grid points
    """
    if points.numel() == 0:
        return torch.zeros(grid_points.shape[0], device=grid_points.device)
    
    # Convert to numpy for scipy KDE
    points_np = points.detach().cpu().numpy()
    grid_np = grid_points.detach().cpu().numpy()
    
    try:
        # Perform KDE
        kde = gaussian_kde(points_np.T, bw_method=bandwidth)
        density_np = kde(grid_np.T)
        
        # Convert back to torch
        density = torch.from_numpy(density_np).float().to(grid_points.device)
        
    except np.linalg.LinAlgError:
        # Fallback to uniform density if KDE fails
        density = torch.ones(grid_points.shape[0], device=grid_points.device) * 0.1
    
    return density


def spatial_interpolation(density_values: torch.Tensor,
                         source_resolution: Tuple[int, int],
                         target_resolution: Tuple[int, int]) -> torch.Tensor:
    """
    Interpolate density values to target resolution using bilinear interpolation.
    
    Args:
        density_values: [H_src * W_src] flat density values
        source_resolution: (H_src, W_src)
        target_resolution: (H_tgt, W_tgt)
        
    Returns:
        interpolated: [H_tgt, W_tgt] interpolated density map
    """
    H_src, W_src = source_resolution
    H_tgt, W_tgt = target_resolution
    
    # Reshape to 2D
    density_2d = density_values.view(H_src, W_src)
    
    # Add batch and channel dimensions for interpolation
    density_4d = density_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, H_src, W_src]
    
    # Bilinear interpolation
    interpolated = F.interpolate(
        density_4d, 
        size=(H_tgt, W_tgt), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Remove batch and channel dimensions
    return interpolated.squeeze(0).squeeze(0)  # [H_tgt, W_tgt]


def create_spatial_grid(height: int, width: int, 
                       device: torch.device,
                       normalize: bool = True) -> torch.Tensor:
    """
    Create spatial coordinate grid for density computation.
    
    Args:
        height: Grid height
        width: Grid width  
        device: Torch device
        normalize: Whether to normalize coordinates to [-1, 1]
        
    Returns:
        grid: [H*W, 2] flattened coordinate grid
    """
    # Create coordinate grids
    y_coords = torch.arange(height, dtype=torch.float32, device=device)
    x_coords = torch.arange(width, dtype=torch.float32, device=device)
    
    if normalize:
        y_coords = (y_coords / (height - 1)) * 2 - 1  # Normalize to [-1, 1]
        x_coords = (x_coords / (width - 1)) * 2 - 1
    
    # Create meshgrid and flatten
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_points = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
    grid_flat = grid_points.view(-1, 2)  # [H*W, 2]
    
    return grid_flat


def tensor_summary(tensor: torch.Tensor, name: str = "tensor") -> str:
    """
    Generate summary statistics for a tensor (useful for debugging).
    
    Args:
        tensor: Input tensor
        name: Name for the tensor
        
    Returns:
        summary: String with tensor statistics
    """
    return (f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, "
           f"device={tensor.device}, min={tensor.min().item():.4f}, "
           f"max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}")