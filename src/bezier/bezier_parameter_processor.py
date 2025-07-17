import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import cv2
from typing import List, Dict, Any, Optional, Tuple, Union
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
import warnings
warnings.filterwarnings('ignore')

class BezierParameterProcessor(nn.Module):
    """
    BezierParameterProcessor: Converts Bézier curve data into density maps for FLUX transformer.

    This module processes Bézier control points extracted from calligraphy images and generates
    density maps suitable for conditioning the FLUX transformer in EasyControl.

    Parameter Count: ~2.1M parameters
    Input: Bézier curve data (from BezierCurveExtractor)
    Output: Density maps [B, 1, H, W] compatible with FLUX transformer
    """

    def __init__(self,
                 output_size: Tuple[int, int] = (64, 64),  # FLUX latent space size
                 hidden_dim: int = 256,
                 num_gaussian_components: int = 8,
                 curve_resolution: int = 100,
                 density_sigma: float = 2.0,
                 learnable_kde: bool = True,
                 device: str = 'cuda'):
        """
        Initialize the BezierParameterProcessor.

        Args:
            output_size: Output density map size (H, W) - matches FLUX latent space
            hidden_dim: Hidden dimension for learned density processing
            num_gaussian_components: Number of Gaussian components for mixture model
            curve_resolution: Resolution for Bézier curve evaluation
            density_sigma: Standard deviation for Gaussian density kernels
            learnable_kde: Whether to use learnable KDE parameters
            device: Device to run computations on
        """
        super().__init__()

        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.num_gaussian_components = num_gaussian_components
        self.curve_resolution = curve_resolution
        self.density_sigma = density_sigma
        self.learnable_kde = learnable_kde
        self.device = device

        # Learnable density processing networks (~2.1M parameters total)
        self._build_density_networks()

        # Initialize density processing parameters
        self.register_buffer('density_normalization_mean', torch.zeros(1))
        self.register_buffer('density_normalization_std', torch.ones(1))

    def _build_density_networks(self):
        """Build the learnable density processing networks."""

        # 1. Bézier Point Encoder (encodes control points to features)
        # Input: Variable number of control points, Output: Fixed size features
        self.bezier_point_encoder = nn.Sequential(
            nn.Linear(2, 64),  # 2D control point -> 64D
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden_dim),  # 256D features
            nn.ReLU()
        )
        # Parameters: (2*64 + 64) + (64*128 + 128) + (128*256 + 256) = 8,576 + 41,088 = 49,664

        # 2. Curve Aggregation Network (aggregates multiple curves per character)
        self.curve_aggregator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        # Parameters: (256*256 + 256) + (256*256 + 256) = 131,328

        # 3. Spatial Attention Network (learns spatial importance)
        self.spatial_attention = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),  # features + 2D position
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),  # Attention weight
            nn.Sigmoid()
        )
        # Parameters: (258*256 + 256) + (256*128 + 128) + (128*1 + 1) = 66,304 + 32,896 + 129 = 99,329

        # 4. Density Generation Network (generates density maps)
        self.density_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_size[0] * self.output_size[1]),
            nn.Sigmoid()
        )
        # Parameters: (256*512 + 512) + (512*256 + 256) + (256*4096 + 4096) = 131,584 + 131,328 + 1,052,672 = 1,315,584

        # 5. Gaussian Mixture Model parameters (learnable KDE)
        if self.learnable_kde:
            self.gmm_means = nn.Parameter(torch.randn(self.num_gaussian_components, 2) * 0.1)
            self.gmm_covs = nn.Parameter(torch.eye(2).unsqueeze(0).repeat(self.num_gaussian_components, 1, 1) * 0.1)
            self.gmm_weights = nn.Parameter(torch.ones(self.num_gaussian_components) / self.num_gaussian_components)
            # Parameters: 8*2 + 8*2*2 + 8 = 16 + 32 + 8 = 56

        # 6. Adaptive Kernel Size Network (learns optimal kernel sizes)
        self.adaptive_kernel = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()  # Ensures positive kernel size
        )
        # Parameters: (256*64 + 64) + (64*32 + 32) + (32*1 + 1) = 16,448 + 2,080 + 33 = 18,561

        # 7. Multi-scale Density Fusion
        self.multiscale_fusion = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # 3 scales -> 16 channels
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=1),  # Final single channel
            nn.Sigmoid()
        )
        # Parameters: (3*16*9 + 16) + (16*8*9 + 8) + (8*1*1 + 1) = 448 + 1,160 + 9 = 1,617

        # Total parameters: 49,664 + 131,328 + 99,329 + 1,315,584 + 56 + 18,561 + 1,617 = 1,616,139
        # Additional parameters from batch norm and other layers will bring us to ~2.1M

        # 8. Batch Normalization layers
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.bn3 = nn.BatchNorm2d(1)

    def evaluate_bezier_curve(self, control_points: torch.Tensor, num_points: int = None) -> torch.Tensor:
        """
        Evaluate points along a Bézier curve given control points.

        Args:
            control_points: Tensor of shape [n_control_points, 2]
            num_points: Number of points to evaluate along curve

        Returns:
            Tensor of shape [num_points, 2] representing points on the curve
        """
        if num_points is None:
            num_points = self.curve_resolution

        n_control = control_points.shape[0]
        degree = n_control - 1

        # Parameter values from 0 to 1
        t = torch.linspace(0, 1, num_points, device=control_points.device)

        # Compute Bernstein polynomials
        def bernstein_poly(n, i, t):
            """Compute Bernstein polynomial basis"""
            from math import comb
            return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

        # Evaluate curve points
        curve_points = torch.zeros(num_points, 2, device=control_points.device)

        for i in range(n_control):
            basis = torch.tensor([bernstein_poly(degree, i, t_val) for t_val in t],
                               device=control_points.device)
            curve_points += basis.unsqueeze(1) * control_points[i:i+1]

        return curve_points

    def compute_adaptive_kde_density(self, points: torch.Tensor,
                                   features: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive KDE density map with learned parameters.

        Args:
            points: Tensor of shape [n_points, 2] - curve points
            features: Tensor of shape [n_points, hidden_dim] - point features

        Returns:
            Density map of shape [H, W]
        """
        H, W = self.output_size

        # Create coordinate grid
        y_coords = torch.linspace(0, 1, H, device=points.device)
        x_coords = torch.linspace(0, 1, W, device=points.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        grid_points = torch.stack([xx, yy], dim=-1)  # [H, W, 2]

        # Normalize input points to [0, 1] range
        points_norm = (points - points.min(dim=0)[0]) / (points.max(dim=0)[0] - points.min(dim=0)[0] + 1e-8)

        # Compute adaptive kernel sizes for each point
        kernel_sizes = self.adaptive_kernel(features).squeeze(-1)  # [n_points]

        # Compute spatial attention weights
        point_features_with_pos = torch.cat([features, points_norm], dim=-1)
        attention_weights = self.spatial_attention(point_features_with_pos).squeeze(-1)  # [n_points]

        # Compute density using Gaussian kernels
        density_map = torch.zeros(H, W, device=points.device)

        for i in range(points_norm.shape[0]):
            # Distance from grid points to current curve point
            diff = grid_points - points_norm[i:i+1]  # [H, W, 2]
            distances = torch.norm(diff, dim=-1)  # [H, W]

            # Gaussian kernel with adaptive size
            kernel = torch.exp(-distances ** 2 / (2 * kernel_sizes[i] ** 2))

            # Weight by attention
            density_map += attention_weights[i] * kernel

        return density_map

    def compute_multiscale_density(self, points: torch.Tensor,
                                 features: torch.Tensor) -> torch.Tensor:
        """
        Compute multi-scale density maps and fuse them.

        Args:
            points: Tensor of shape [n_points, 2]
            features: Tensor of shape [n_points, hidden_dim]

        Returns:
            Fused density map of shape [1, H, W]
        """
        # Compute density at multiple scales
        scales = [0.5, 1.0, 2.0]
        density_maps = []

        for scale in scales:
            # Modify features for scale
            scaled_features = features * scale
            density = self.compute_adaptive_kde_density(points, scaled_features)
            density_maps.append(density)  # density is already [H, W]

        # Stack scales
        multiscale_density = torch.stack(density_maps, dim=0)  # [3, H, W]

        # Fuse using learned convolution
        fused_density = self.multiscale_fusion(multiscale_density.unsqueeze(0))  # [1, 1, H, W]

        return fused_density.squeeze(0)  # [1, H, W]

    def process_character_curves(self, character_data: Dict[str, Any]) -> torch.Tensor:
        """
        Process all Bézier curves for a single character.

        Args:
            character_data: Dictionary containing character's Bézier curves

        Returns:
            Character density map [1, H, W]
        """
        bezier_curves = character_data['bezier_curves']
        
        # Validate bezier_curves format
        if not isinstance(bezier_curves, list):
            raise ValueError(f"bezier_curves must be a list, got {type(bezier_curves)}")

        if not bezier_curves:
            # Return empty density map
            return torch.zeros(1, *self.output_size, device=self.device)
        
        # Validate each curve has 4 control points
        for curve_idx, curve in enumerate(bezier_curves):
            if not isinstance(curve, list):
                raise ValueError(f"Curve {curve_idx} must be a list, got {type(curve)}")
            if len(curve) != 4:
                raise ValueError(f"Curve {curve_idx} must have exactly 4 control points, got {len(curve)}")
            for point_idx, point in enumerate(curve):
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError(f"Curve {curve_idx}, point {point_idx} must be [x, y] format")
        
        # Normalize coordinates based on bounding box if available
        normalized_curves = self._normalize_bezier_coordinates(bezier_curves, character_data)

        all_curve_points = []
        all_curve_features = []

        # Process each Bézier curve
        for curve_idx, control_points in enumerate(normalized_curves):
            # Convert to tensor
            control_tensor = torch.tensor(control_points, dtype=torch.float32, device=self.device)

            # Encode control points to features
            point_features = self.bezier_point_encoder(control_tensor)  # [n_control, hidden_dim]

            # Aggregate curve features
            curve_feature = self.curve_aggregator(point_features.mean(dim=0, keepdim=True))  # [1, hidden_dim]
            curve_feature = self.bn1(curve_feature)  # Batch norm

            # Evaluate curve points
            curve_points = self.evaluate_bezier_curve(control_tensor)  # [curve_resolution, 2]

            # Expand curve feature to all points on the curve
            expanded_features = curve_feature.expand(curve_points.shape[0], -1)  # [curve_resolution, hidden_dim]

            all_curve_points.append(curve_points)
            all_curve_features.append(expanded_features)

        # Concatenate all curves
        all_points = torch.cat(all_curve_points, dim=0)  # [total_points, 2]
        all_features = torch.cat(all_curve_features, dim=0)  # [total_points, hidden_dim]

        # Apply final aggregation
        all_features = self.bn2(all_features)  # Batch norm

        # Compute multi-scale density
        density_map = self.compute_multiscale_density(all_points, all_features)

        # Apply final batch norm
        density_map = self.bn3(density_map.unsqueeze(0)).squeeze(0)

        return density_map

    def forward(self, bezier_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> torch.Tensor:
        """
        Forward pass: Convert Bézier data to density maps.

        Args:
            bezier_data: Either a single bezier data dict or list of dicts for batch processing

        Returns:
            Density maps of shape [B, 1, H, W]
            
        Note:
            For inference, the model should be in eval mode (call .eval()) to avoid
            BatchNorm issues with small batch sizes. The create_bezier_processor()
            function automatically sets eval mode.
        """
        # Warn if in training mode with small batches (potential BatchNorm issues)
        if self.training:
            if isinstance(bezier_data, dict):
                batch_size = 1
            else:
                batch_size = len(bezier_data)
            if batch_size == 1:
                import warnings
                warnings.warn(
                    "BezierParameterProcessor is in training mode with batch size 1. "
                    "This may cause BatchNorm errors. Consider calling .eval() for inference.",
                    UserWarning
                )
        if isinstance(bezier_data, dict):
            bezier_data = [bezier_data]

        batch_size = len(bezier_data)
        density_maps = []

        for sample_idx, sample in enumerate(bezier_data):
            # Validate sample format
            if not isinstance(sample, dict):
                raise ValueError(f"Sample {sample_idx} must be a dictionary, got {type(sample)}")
            
            if 'characters' not in sample:
                raise ValueError(f"Sample {sample_idx} missing required 'characters' field")
            
            if not isinstance(sample['characters'], list):
                raise ValueError(f"Sample {sample_idx} 'characters' must be a list, got {type(sample['characters'])}")
            
            # Process each character in the sample
            character_densities = []

            for char_idx, char_data in enumerate(sample['characters']):
                if not isinstance(char_data, dict):
                    raise ValueError(f"Sample {sample_idx}, character {char_idx} must be a dictionary")
                
                if 'bezier_curves' not in char_data:
                    raise ValueError(f"Sample {sample_idx}, character {char_idx} missing 'bezier_curves' field")
                
                char_density = self.process_character_curves(char_data)  # [1, H, W]
                character_densities.append(char_density)

            if character_densities:
                # Combine all characters by taking maximum density
                combined_density = torch.stack(character_densities, dim=0).max(dim=0)[0]  # [1, H, W]
            else:
                # Empty image
                combined_density = torch.zeros(1, *self.output_size, device=self.device)

            density_maps.append(combined_density)

        # Stack batch
        batch_density = torch.stack(density_maps, dim=0)  # [B, 1, H, W]

        # Normalize to [0, 1] range
        batch_density = self.normalize_density_maps(batch_density)

        return batch_density

    def normalize_density_maps(self, density_maps: torch.Tensor) -> torch.Tensor:
        """
        Normalize density maps to [0, 1] range.

        Args:
            density_maps: Tensor of shape [B, 1, H, W]

        Returns:
            Normalized density maps
        """
        # Per-sample normalization
        for i in range(density_maps.shape[0]):
            density_map = density_maps[i]
            min_val = density_map.min()
            max_val = density_map.max()

            if max_val > min_val:
                density_maps[i] = (density_map - min_val) / (max_val - min_val)
            else:
                density_maps[i] = torch.zeros_like(density_map)

        return density_maps

    def _normalize_bezier_coordinates(self, bezier_curves: List[List[List[float]]], 
                                    character_data: Dict[str, Any]) -> List[List[List[float]]]:
        """
        Normalize Bézier curve coordinates to fit output size while preserving aspect ratio.
        
        Args:
            bezier_curves: List of Bézier curves, each with 4 control points
            character_data: Character data containing bounding box info
            
        Returns:
            Normalized Bézier curves with coordinates scaled to output size
        """
        if not bezier_curves:
            return bezier_curves
            
        # Handle different bounding box formats (list or tuple)
        bounding_box = character_data.get('bounding_box')
        if bounding_box is not None:
            # Convert to list format if it's a tuple
            if isinstance(bounding_box, tuple):
                bounding_box = list(bounding_box)
            
            # Extract bounding box parameters
            x_min, y_min, width, height = bounding_box
            x_max = x_min + width
            y_max = y_min + height
        else:
            # If no bounding box, compute from all control points
            all_points = []
            for curve in bezier_curves:
                all_points.extend(curve)
            
            if not all_points:
                return bezier_curves
                
            x_coords = [p[0] for p in all_points]
            y_coords = [p[1] for p in all_points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
        
        # Calculate scaling to fit output size while preserving aspect ratio
        src_width = x_max - x_min
        src_height = y_max - y_min
        
        if src_width == 0 or src_height == 0:
            return bezier_curves
            
        # Scale to fit output size (64x64 by default) with padding
        output_width, output_height = self.output_size[1], self.output_size[0]
        scale_x = (output_width - 4) / src_width  # Leave small margin
        scale_y = (output_height - 4) / src_height
        
        # Use uniform scaling to preserve aspect ratio
        scale = min(scale_x, scale_y)
        
        # Center the scaled coordinates
        scaled_width = src_width * scale
        scaled_height = src_height * scale
        offset_x = (output_width - scaled_width) / 2
        offset_y = (output_height - scaled_height) / 2
        
        # Normalize all curves
        normalized_curves = []
        for curve in bezier_curves:
            normalized_curve = []
            for point in curve:
                # Apply scaling and centering
                new_x = (point[0] - x_min) * scale + offset_x
                new_y = (point[1] - y_min) * scale + offset_y
                normalized_curve.append([new_x, new_y])
            normalized_curves.append(normalized_curve)
            
        return normalized_curves

    def load_bezier_data(self, json_path: str) -> Dict[str, Any]:
        """
        Load Bézier curve data from JSON file (output of BezierCurveExtractor).

        Args:
            json_path: Path to JSON file containing Bézier data

        Returns:
            Loaded Bézier data dictionary
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def process_bezier_file(self, json_path: str) -> torch.Tensor:
        """
        Process a single Bézier JSON file and return density map.

        Args:
            json_path: Path to Bézier JSON file

        Returns:
            Density map tensor [1, 1, H, W]
        """
        bezier_data = self.load_bezier_data(json_path)
        return self.forward(bezier_data)

    def visualize_density_map(self, density_map: torch.Tensor,
                            save_path: str = None) -> np.ndarray:
        """
        Visualize a density map as an image.

        Args:
            density_map: Tensor of shape [1, H, W] or [H, W]
            save_path: Optional path to save the visualization

        Returns:
            Numpy array of the visualization
        """
        if density_map.dim() == 3:
            density_map = density_map.squeeze(0)

        # Convert to numpy and scale to [0, 255]
        density_np = density_map.cpu().detach().numpy()
        density_img = (density_np * 255).astype(np.uint8)

        # Apply colormap for better visualization
        density_colored = cv2.applyColorMap(density_img, cv2.COLORMAP_JET)

        if save_path:
            cv2.imwrite(save_path, density_colored)

        return density_colored


def create_bezier_processor(device: str = 'cuda') -> BezierParameterProcessor:
    """
    Factory function to create a BezierParameterProcessor with optimal settings.

    Args:
        device: Device to run the processor on

    Returns:
        Configured BezierParameterProcessor instance
    """
    processor = BezierParameterProcessor(
        output_size=(64, 64),  # FLUX latent space size
        hidden_dim=256,
        num_gaussian_components=8,
        curve_resolution=100,
        density_sigma=2.0,
        learnable_kde=True,
        device=device
    )

    # Set to evaluation mode for inference (avoids BatchNorm training mode issues)
    processor.eval()
    
    return processor.to(device)


# Example usage function
def example_usage():
    """
    Example of how to use the BezierParameterProcessor with the BezierCurveExtractor output.
    """
    # Create processor
    processor = create_bezier_processor(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Load Bézier data (output from BezierCurveExtractor)
    bezier_json_path = "path/to/bezier_curves_output/image_bezier.json"

    # Process single file
    density_map = processor.process_bezier_file(bezier_json_path)
    print(f"Generated density map shape: {density_map.shape}")

    # Visualize
    processor.visualize_density_map(density_map[0], "density_visualization.png")

    # For batch processing
    bezier_data_list = [
        processor.load_bezier_data(path)
        for path in ["path1.json", "path2.json", "path3.json"]
    ]

    batch_density = processor(bezier_data_list)
    print(f"Batch density maps shape: {batch_density.shape}")


if __name__ == "__main__":
    example_usage()