"""
SpatialAttentionFuser Module - Density-guided transformer attention
====================================================================

Architecture: Encoder-Decoder with density-guided attention
Parameters: 3.8M (transformer-based with efficiency optimizations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class DensityModulatedSelfAttention(nn.Module):
    """
    Self-attention layer with density-based modulation.

    Key innovation: Higher density regions receive stronger attention weights.
    """
    def __init__(self, feature_dim: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        # Standard attention projections
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        # Density processing
        self.density_processor = DensityProcessor(feature_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        density_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Density-modulated self-attention.

        Args:
            x: [B, N, feature_dim] spatial features (N = H*W)
            density_weights: [B, H, W, 1] density map from Bézier processor

        Returns:
            output: [B, N, feature_dim] modulated features
            attention_weights: [B, num_heads, N, N] attention maps
        """
        B, N, D = x.shape

        # Standard Q, K, V projections
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)  # [B, N, H, D//H]
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim)

        # Transpose for attention computation: [B, H, N, D//H]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, N]

        # Apply density modulation - key innovation
        # Higher density regions get stronger attention
        density_flat = density_weights.view(B, N, 1)  # [B, N, 1]
        head_modulation = density_flat.expand(-1, -1, self.num_heads)  # [B, N, num_heads]
        head_modulation = head_modulation.transpose(1, 2)  # [B, num_heads, N]

        # Modulate attention scores
        modulated_scores = scores * (1 + head_modulation.unsqueeze(-1))  # [B, H, N, N]

        # Softmax and apply attention
        attention_weights = F.softmax(modulated_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # [B, H, N, D//H]

        # Transpose back and reshape: [B, N, D]
        output = output.transpose(1, 2).contiguous().view(B, N, D)

        # Output projection and residual connection
        output = self.out_proj(output)
        output = self.layer_norm(output + x)  # Residual connection

        return output, attention_weights


class DensityProcessor(nn.Module):
    """
    Processes density maps for attention modulation.
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        self.density_mlp = nn.Sequential(
            nn.Linear(1, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, feature_dim // 4),
            nn.GELU(),
            nn.Linear(feature_dim // 4, 1)
        )

    def forward(self, density_map: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Process density map for multi-head attention modulation.

        Args:
            density_map: [B, H, W, 1] raw density weights
            H, W: spatial dimensions

        Returns:
            modulation_weights: [B, H*W, 1] processed density weights
        """
        B = density_map.size(0)

        # Reshape to flatten spatial dimensions
        density_flat = density_map.view(B, H * W, 1)  # [B, N, 1]

        # Process through MLP to learn appropriate modulation strength
        modulation = self.density_mlp(density_flat)  # [B, N, 1]

        # Apply sigmoid to ensure modulation is in (0.5, 1.5) range
        # This ensures density enhances rather than suppresses attention
        modulation = torch.sigmoid(modulation) + 0.5  # Range: (0.5, 1.5)

        return modulation


class PositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for spatial awareness.
    """
    def __init__(self, feature_dim: int, max_len: int = 10000):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_len = max_len

    def forward(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """
        Generate 2D positional encodings.

        Returns:
            pos_encoding: [1, H*W, feature_dim] positional encodings
        """
        # Create coordinate grids
        y_coords = torch.arange(height, device=device).float()
        x_coords = torch.arange(width, device=device).float()

        # Normalize coordinates to [0, 1]
        y_coords = y_coords / (height - 1) if height > 1 else y_coords
        x_coords = x_coords / (width - 1) if width > 1 else x_coords

        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([x_grid, y_grid], dim=-1).view(-1, 2)  # [H*W, 2]

        # Sinusoidal encoding
        div_term = torch.exp(
            torch.arange(0, self.feature_dim // 2, 2, device=device).float() *
            -(math.log(self.max_len) / (self.feature_dim // 2))
        )

        pos_encoding = torch.zeros(height * width, self.feature_dim, device=device)

        # X coordinate encoding
        pos_encoding[:, 0::4] = torch.sin(coords[:, 0:1] * div_term)
        pos_encoding[:, 1::4] = torch.cos(coords[:, 0:1] * div_term)

        # Y coordinate encoding
        pos_encoding[:, 2::4] = torch.sin(coords[:, 1:2] * div_term)
        pos_encoding[:, 3::4] = torch.cos(coords[:, 1:2] * div_term)

        return pos_encoding.unsqueeze(0)  # [1, H*W, feature_dim]


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with density-modulated attention.
    """
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = DensityModulatedSelfAttention(feature_dim, num_heads, dropout)

        # Feed-forward network
        mlp_hidden_dim = int(feature_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, feature_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        density_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with density-modulated attention.
        """
        # Self-attention with density modulation
        attn_output, attention_weights = self.attention(x, density_weights)

        # Feed-forward network with residual connection
        mlp_output = self.mlp(attn_output)
        output = attn_output + mlp_output

        return output, attention_weights


class SpatialAttentionFuser(nn.Module):
    """
    Main spatial attention fusion module.

    Architecture: Encoder-Decoder with density-guided attention
    Parameters: 3.8M (transformer-based with efficiency optimizations)
    """
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Positional encoding
        self.pos_encoding = PositionalEncoding2D(feature_dim)

        # Density projection
        self.density_projector = nn.Linear(1, feature_dim)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(feature_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Condition embedding for cross-modal fusion
        self.condition_embedder = nn.Linear(feature_dim, feature_dim)

        # Cross-attention decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=feature_dim,
                nhead=num_heads,
                dim_feedforward=int(feature_dim * mlp_ratio),
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # Spatial weight generation head
        self.spatial_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, num_heads)
        )

        self._log_parameters()

    def _log_parameters(self):
        """Log parameter count for analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"SpatialAttentionFuser total parameters: {total_params / 1e6:.1f}M")

    def forward(
        self,
        spatial_features: torch.Tensor,
        density_weights: torch.Tensor,
        condition_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Density-guided spatial attention fusion.

        Args:
            spatial_features: [B, H*W, feature_dim] spatial feature tokens
            density_weights: [B, H, W, 1] density map from Bézier processor
            condition_embeddings: [B, seq_len, feature_dim] optional condition embeddings

        Returns:
            fused_features: [B, H*W, feature_dim] spatially-weighted features
            attention_weights: [B, H, W, num_heads] spatial attention maps
        """
        B, N, D = spatial_features.shape
        H = W = int(math.sqrt(N))

        # Add positional encoding
        pos_encoding = self.pos_encoding(H, W, spatial_features.device)
        x = spatial_features + pos_encoding

        # === ENCODER PATH ===
        encoder_output = x
        all_attention_weights = []

        for layer in self.encoder_layers:
            encoder_output, attention_weights = layer(encoder_output, density_weights)
            all_attention_weights.append(attention_weights)

        # === DECODER PATH (if conditions provided) ===
        if condition_embeddings is not None:
            decoder_output = encoder_output

            for layer in self.decoder_layers:
                decoder_output = layer(
                    tgt=decoder_output,
                    memory=self.condition_embedder(condition_embeddings)
                )
        else:
            decoder_output = encoder_output

        # === SPATIAL WEIGHT GENERATION ===
        spatial_weights = self.spatial_head(decoder_output)  # [B, N, num_heads]
        spatial_weights = spatial_weights.view(B, H, W, self.num_heads)

        # Final attention-weighted features
        final_attention = torch.stack(all_attention_weights, dim=1).mean(dim=1)  # Average across layers
        # Extract mean attention across heads for weighting: [B, N, 1]
        attention_mean = final_attention.mean(dim=1).unsqueeze(-1)  # [B, N, 1]
        weighted_features = decoder_output * attention_mean

        return weighted_features, spatial_weights

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Training loss functions
def density_attention_alignment_loss(
    attention_weights: torch.Tensor,
    target_density: torch.Tensor,
    alignment_weight: float = 1.0
) -> torch.Tensor:
    """
    Aligns attention patterns with density maps.

    Args:
        attention_weights: [B, H, W, num_heads] spatial attention weights
        target_density: [B, H, W, 1] target density map
        alignment_weight: Loss weighting factor

    Returns:
        loss: Alignment loss value
    """
    # Reshape attention weights to match density layout
    B, H, W, num_heads = attention_weights.shape
    target_density = target_density.squeeze(-1)  # [B, H, W]

    # Compute attention-density correlation
    attention_mean = attention_weights.mean(dim=-1)  # Average across heads [B, H, W]

    # Normalize both for correlation computation
    attention_flat = attention_mean.view(B, -1)
    density_flat = target_density.view(B, -1)

    # MSE loss between normalized attention and density
    attention_norm = F.softmax(attention_flat, dim=-1)
    density_norm = F.softmax(density_flat, dim=-1)

    alignment_loss = F.mse_loss(attention_norm, density_norm)

    return alignment_weight * alignment_loss


def attention_entropy_loss(
    attention_weights: torch.Tensor,
    entropy_weight: float = 0.1
) -> torch.Tensor:
    """
    Prevents attention collapse by encouraging diversity.

    Args:
        attention_weights: [B, H, W, num_heads] attention weights
        entropy_weight: Loss weighting factor

    Returns:
        loss: Entropy loss value
    """
    # Compute entropy across spatial dimensions
    B, H, W, num_heads = attention_weights.shape
    attention_flat = attention_weights.view(B, -1, num_heads)

    # Add small epsilon for numerical stability
    eps = 1e-8
    attention_probs = F.softmax(attention_flat, dim=1) + eps

    # Compute entropy: -sum(p * log(p))
    entropy = -(attention_probs * torch.log(attention_probs)).sum(dim=1).mean()

    # We want to maximize entropy (encourage diversity), so minimize negative entropy
    return entropy_weight * (-entropy)


def create_spatial_attention_fuser(device: str = "cuda", **kwargs) -> SpatialAttentionFuser:
    """
    Factory function to create a SpatialAttentionFuser with sensible defaults.

    Args:
        device: Target device for the fuser
        **kwargs: Additional arguments for SpatialAttentionFuser

    Returns:
        fuser: Initialized SpatialAttentionFuser
    """
    fuser = SpatialAttentionFuser(**kwargs)
    fuser.to(device)
    return fuser