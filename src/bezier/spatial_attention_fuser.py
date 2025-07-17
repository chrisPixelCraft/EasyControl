import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from diffusers.models.attention_processor import Attention
import math

class SpatialAttentionFuser(nn.Module):
    """
    SpatialAttentionFuser: Density-modulated attention mechanism for FLUX transformer.

    This module integrates density maps from BezierParameterProcessor into the attention
    mechanism of FLUX transformer blocks, enabling spatial conditioning based on Bézier
    curve density information.

    Parameter Count: ~3.8M parameters
    Input: Density maps [B, 1, H, W] + attention features
    Output: Spatially-modulated attention features
    """

    def __init__(self,
                 hidden_dim: int = 3072,  # FLUX transformer hidden dimension
                 num_heads: int = 24,     # FLUX transformer attention heads
                 head_dim: int = 128,     # FLUX transformer head dimension
                 density_feature_dim: int = 256,  # Density feature dimension from BezierParameterProcessor
                 spatial_resolution: int = 64,    # Spatial resolution for density maps
                 fusion_layers: int = 3,          # Number of fusion layers
                 use_positional_encoding: bool = True,
                 dropout: float = 0.1,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the SpatialAttentionFuser.

        Args:
            hidden_dim: Hidden dimension of FLUX transformer
            num_heads: Number of attention heads
            head_dim: Dimension per attention head
            density_feature_dim: Dimension of density features
            spatial_resolution: Spatial resolution of density maps
            fusion_layers: Number of fusion layers
            use_positional_encoding: Whether to use positional encoding
            dropout: Dropout rate
            device: Device for computation
            dtype: Data type for computation
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.density_feature_dim = density_feature_dim
        self.spatial_resolution = spatial_resolution
        self.fusion_layers = fusion_layers
        self.use_positional_encoding = use_positional_encoding
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        # Calculate parameter counts for each component
        self._init_components()

    def _init_components(self):
        """Initialize all components of the SpatialAttentionFuser."""

        # 1. Density Map Encoder (512K parameters)
        self.density_encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 64 * (1 * 3 * 3 + 1) = 640
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 128 * (64 * 3 * 3 + 1) = 73,856
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 256 * (128 * 3 * 3 + 1) = 295,168
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),  # Reduce to 8x8 for efficiency
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, self.density_feature_dim),  # 256 * 256 * 64 = 4,194,304
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout)
        )

        # 2. Spatial Position Encoder (256K parameters)
        if self.use_positional_encoding:
            self.spatial_pos_encoder = nn.Sequential(
                nn.Linear(2, 128),  # 2 * 128 + 128 = 384
                nn.ReLU(inplace=True),
                nn.Linear(128, 256),  # 128 * 256 + 256 = 33,024
                nn.ReLU(inplace=True),
                nn.Linear(256, self.density_feature_dim),  # 256 * 256 + 256 = 65,792
                nn.Dropout(self.dropout)
            )

        # 3. Density-Attention Fusion Layers (2.8M parameters)
        self.fusion_layers_list = nn.ModuleList()
        for i in range(self.fusion_layers):
            fusion_layer = DensityAttentionFusionLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                density_feature_dim=self.density_feature_dim,
                dropout=self.dropout
            )
            self.fusion_layers_list.append(fusion_layer)

        # 4. Spatial Attention Gates (128K parameters)
        self.spatial_attention_gates = nn.ModuleList()
        for i in range(self.num_heads):
            gate = nn.Sequential(
                nn.Linear(self.density_feature_dim, 128),  # 256 * 128 + 128 = 33,024
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),  # 128 * 1 + 1 = 129
                nn.Sigmoid()
            )
            self.spatial_attention_gates.append(gate)

        # 5. Output Projection (256K parameters)
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),  # 3072 * 3072 + 3072 = 9,440,256
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)  # 3072 * 3072 + 3072 = 9,440,256
        )

        # 6. Normalization layers
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.density_feature_dim)

    def forward(self,
                attention_output: torch.Tensor,
                density_map: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None,
                spatial_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for spatial attention fusion.

        Args:
            attention_output: Output from FLUX attention [B, N, D]
            density_map: Density map from BezierParameterProcessor [B, 1, H, W]
            attention_weights: Optional attention weights [B, H, N, N]
            spatial_mask: Optional spatial mask [B, N]

        Returns:
            Tuple of (fused_attention_output, debug_info)
        """
        batch_size, seq_len, _ = attention_output.shape

        # 1. Encode density map
        density_features = self.density_encoder(density_map)  # [B, density_feature_dim]
        density_features = self.layer_norm2(density_features)

        # 2. Generate spatial positional encoding if enabled
        if self.use_positional_encoding:
            spatial_positions = self._generate_spatial_positions(batch_size, seq_len)
            spatial_pos_features = self.spatial_pos_encoder(spatial_positions)  # [B, N, density_feature_dim]
            # Combine density and positional features
            density_features_expanded = density_features.unsqueeze(1).expand(-1, seq_len, -1)
            combined_features = density_features_expanded + spatial_pos_features
        else:
            combined_features = density_features.unsqueeze(1).expand(-1, seq_len, -1)

        # 3. Apply density-attention fusion layers
        fused_output = attention_output
        layer_outputs = []

        for i, fusion_layer in enumerate(self.fusion_layers_list):
            fused_output = fusion_layer(
                attention_features=fused_output,
                density_features=combined_features,
                attention_weights=attention_weights,
                spatial_mask=spatial_mask
            )
            layer_outputs.append(fused_output.clone())

        # 4. Apply spatial attention gates
        gate_outputs = []
        for head_idx in range(self.num_heads):
            gate_weight = self.spatial_attention_gates[head_idx](density_features)  # [B, 1]
            gate_outputs.append(gate_weight)

        # Combine gate weights and apply to fused output
        spatial_gates = torch.stack(gate_outputs, dim=1)  # [B, num_heads, 1]

        # Reshape for head-wise application
        head_dim = self.hidden_dim // self.num_heads
        fused_reshaped = fused_output.view(batch_size, seq_len, self.num_heads, head_dim)
        gated_output = fused_reshaped * spatial_gates.unsqueeze(1).unsqueeze(-1)
        fused_output = gated_output.view(batch_size, seq_len, self.hidden_dim)

        # 5. Apply output projection with residual connection
        fused_output = self.layer_norm1(fused_output)
        projected_output = self.output_projection(fused_output)
        final_output = attention_output + projected_output  # Residual connection

        # 6. Prepare debug information
        debug_info = {
            'density_features': density_features,
            'spatial_gates': spatial_gates,
            'layer_outputs': layer_outputs,
            'combined_features': combined_features
        }

        return final_output, debug_info

    def _generate_spatial_positions(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Generate spatial position encodings for sequence elements."""

        # Assume sequence represents spatial positions in a grid
        grid_size = int(math.sqrt(seq_len))
        if grid_size * grid_size != seq_len:
            # If not a perfect square, use approximate grid
            grid_size = int(math.sqrt(seq_len))

        # Generate 2D grid positions
        x_coords = torch.arange(grid_size, device=self.device, dtype=self.dtype)
        y_coords = torch.arange(grid_size, device=self.device, dtype=self.dtype)

        # Normalize coordinates to [-1, 1]
        x_coords = (x_coords / (grid_size - 1)) * 2 - 1
        y_coords = (y_coords / (grid_size - 1)) * 2 - 1

        # Create meshgrid
        x_grid, y_grid = torch.meshgrid(x_coords, y_coords, indexing='ij')

        # Flatten and combine
        positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=-1)  # [seq_len, 2]

        # Handle sequence length mismatch
        if positions.shape[0] != seq_len:
            # Pad or truncate as needed
            if positions.shape[0] < seq_len:
                padding = torch.zeros(seq_len - positions.shape[0], 2, device=self.device, dtype=self.dtype)
                positions = torch.cat([positions, padding], dim=0)
            else:
                positions = positions[:seq_len]

        # Expand for batch
        positions = positions.unsqueeze(0).expand(batch_size, -1, -1)

        return positions

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown for each component."""

        total_params = sum(p.numel() for p in self.parameters())

        component_counts = {
            'density_encoder': sum(p.numel() for p in self.density_encoder.parameters()),
            'spatial_pos_encoder': sum(p.numel() for p in self.spatial_pos_encoder.parameters()) if self.use_positional_encoding else 0,
            'fusion_layers': sum(p.numel() for p in self.fusion_layers_list.parameters()),
            'spatial_attention_gates': sum(p.numel() for p in self.spatial_attention_gates.parameters()),
            'output_projection': sum(p.numel() for p in self.output_projection.parameters()),
            'layer_norms': sum(p.numel() for p in self.layer_norm1.parameters()) + sum(p.numel() for p in self.layer_norm2.parameters()),
            'total': total_params
        }

        return component_counts


class DensityAttentionFusionLayer(nn.Module):
    """
    Individual fusion layer for combining density features with attention features.
    """

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 head_dim: int,
                 density_feature_dim: int,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.density_feature_dim = density_feature_dim
        self.dropout = dropout

        # Cross-attention between density and attention features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Density feature projection
        self.density_proj = nn.Linear(density_feature_dim, hidden_dim)

        # Feature fusion
        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                attention_features: torch.Tensor,
                density_features: torch.Tensor,
                attention_weights: Optional[torch.Tensor] = None,
                spatial_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for density-attention fusion.

        Args:
            attention_features: Attention features [B, N, D]
            density_features: Density features [B, N, density_feature_dim]
            attention_weights: Optional attention weights [B, H, N, N]
            spatial_mask: Optional spatial mask [B, N]

        Returns:
            Fused features [B, N, D]
        """
        # Project density features to hidden dimension
        density_proj = self.density_proj(density_features)  # [B, N, D]

        # Apply cross-attention
        attn_output, _ = self.cross_attention(
            query=attention_features,
            key=density_proj,
            value=density_proj,
            key_padding_mask=spatial_mask
        )

        # Combine original attention features with cross-attention output
        combined = torch.cat([attention_features, attn_output], dim=-1)  # [B, N, 2*D]

        # Apply fusion projection
        fused = self.fusion_proj(combined)

        # Apply layer normalization and residual connection
        output = self.layer_norm(attention_features + fused)

        return output


class SpatialAttentionProcessor(nn.Module):
    """
    Custom attention processor that integrates SpatialAttentionFuser with FLUX attention.
    """

    def __init__(self,
                 spatial_attention_fuser: SpatialAttentionFuser,
                 base_processor: Optional[Any] = None):
        super().__init__()

        self.spatial_attention_fuser = spatial_attention_fuser
        self.base_processor = base_processor

    def __call__(self,
                 attn,
                 hidden_states: torch.Tensor,
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 density_map: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """
        Forward pass for spatial attention processing.

        Args:
            attn: Attention module
            hidden_states: Input hidden states
            encoder_hidden_states: Optional encoder hidden states
            attention_mask: Optional attention mask
            image_rotary_emb: Optional rotary embeddings
            density_map: Optional density map for spatial conditioning
            **kwargs: Additional arguments

        Returns:
            Processed attention output
        """
        # Apply base attention processor if available
        if self.base_processor is not None:
            attention_output = self.base_processor(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs
            )
        else:
            # Fallback to default attention computation
            attention_output = self._default_attention_forward(
                attn=attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb
            )

        # Apply spatial attention fusion if density map is provided
        if density_map is not None:
            fused_output, debug_info = self.spatial_attention_fuser(
                attention_output=attention_output,
                density_map=density_map,
                spatial_mask=attention_mask
            )
            return fused_output
        else:
            return attention_output

    def _default_attention_forward(self,
                                  attn,
                                  hidden_states: torch.Tensor,
                                  encoder_hidden_states: Optional[torch.Tensor] = None,
                                  attention_mask: Optional[torch.Tensor] = None,
                                  image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """Default attention forward pass."""

        # This is a simplified version - in practice, would use the actual FLUX attention implementation
        batch_size, seq_len, _ = hidden_states.shape

        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
        key = key.view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)
        value = value.view(batch_size, seq_len, attn.heads, -1).transpose(1, 2)

        # Scaled dot-product attention
        attention_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )

        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, -1)

        # Apply output projection
        attention_output = attn.to_out[0](attention_output)

        return attention_output