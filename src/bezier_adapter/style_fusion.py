"""
StyleBezierFusionModule - Style transfer with Bézier density modulation
========================================================================

Integrates style features with spatial control using AdaIN and density guidance.
Architecture: Multi-Head Cross-Attention → Density Modulation → AdaIN → Spatial Fusion
Parameters: ~3.5M (optimized for transformer integration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class StyleBezierFusionModule(nn.Module):
    """
    Multi-modal fusion module combining style transfer with Bézier spatial control.
    
    Architecture: Multi-Head Cross-Attention → Density Modulation → AdaIN → Spatial Fusion
    Parameters: ~3.5M (optimized for transformer integration)
    Critical Placement: After attention layers in transformer decoder for maximum effectiveness
    """
    
    def __init__(
        self,
        feature_dim: int = 3072,  # FluxTransformer inner_dim
        style_dim: int = 768,
        text_dim: int = 768, 
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.style_dim = style_dim
        self.num_heads = num_heads
        
        # === 1. MULTI-HEAD CROSS-ATTENTION ===
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        
        # Style and text projection for cross-attention
        self.style_proj = nn.Linear(style_dim, feature_dim)
        self.text_proj = nn.Linear(text_dim, feature_dim)
        
        # === 2. DENSITY MODULATION LAYER ===
        self.density_proj = nn.Conv1d(1, feature_dim, kernel_size=1)
        self.density_gate = nn.Sigmoid()
        
        # === 3. ADAPTIVE INSTANCE NORMALIZATION (AdaIN) ===
        # Style transfer mechanism
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, feature_dim * 2),  # For mean and std
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim * 2, feature_dim * 2)
        )
        self.instance_norm = nn.InstanceNorm1d(feature_dim, affine=False)
        
        # === 4. SPATIAL FUSION BLOCK ===
        self.fusion_layers = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Layer normalization and residual connections
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
        self._initialize_weights()
        self._log_parameters()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier uniform distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _log_parameters(self):
        """Log parameter count for analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        print(f"StyleBezierFusionModule total parameters: {total_params / 1e6:.1f}M")
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        style_features: torch.Tensor,
        bezier_density: torch.Tensor,
        text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: Multi-modal style fusion with Bézier guidance.
        
        Args:
            spatial_features: [B, N, feature_dim] transformer features
            style_features: [B, style_dim] CLIP-encoded style image features
            bezier_density: [B, 1, H, W] density map from BezierParameterProcessor
            text_features: [B, text_dim] optional text prompt embeddings
            
        Returns:
            fused_features: [B, N, feature_dim] style-fused spatial features
        """
        B, N, D = spatial_features.shape
        
        # === 1. MULTI-HEAD CROSS-ATTENTION FUSION ===
        attended_features = self._cross_modal_attention_fusion(
            spatial_features, style_features, text_features
        )
        
        # === 2. DENSITY MODULATION ===
        modulated_features = self._apply_density_modulation(
            attended_features, bezier_density, N
        )
        
        # === 3. ADAPTIVE INSTANCE NORMALIZATION (AdaIN) ===
        styled_features = self._apply_adain_style_transfer(
            modulated_features, style_features
        )
        
        # === 4. SPATIAL FUSION WITH RESIDUAL ===
        fused_features = self._spatial_fusion_block(
            styled_features, spatial_features
        )
        
        return fused_features
    
    def _cross_modal_attention_fusion(
        self,
        spatial_features: torch.Tensor,
        style_features: torch.Tensor,
        text_features: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse style and text features with spatial queries using cross-attention.
        
        Args:
            spatial_features: [B, N, D] spatial features as queries
            style_features: [B, style_dim] style features
            text_features: [B, text_dim] optional text features
            
        Returns:
            attended_features: [B, N, D] cross-attention output
        """
        B, N, D = spatial_features.shape
        
        # Project style features: [B, 1, D]
        style_projected = self.style_proj(style_features).unsqueeze(1)
        
        # Combine style and text features as keys/values
        if text_features is not None:
            text_projected = self.text_proj(text_features).unsqueeze(1)
            kv_features = torch.cat([style_projected, text_projected], dim=1)  # [B, 2, D]
        else:
            kv_features = style_projected  # [B, 1, D]
        
        # Cross-attention: spatial features attend to style/text
        attended_features, attention_weights = self.cross_attention(
            query=spatial_features,  # [B, N, D]
            key=kv_features,        # [B, 1 or 2, D]
            value=kv_features       # [B, 1 or 2, D]
        )
        
        return attended_features
    
    def _apply_density_modulation(
        self,
        features: torch.Tensor,
        bezier_density: torch.Tensor,
        seq_length: int
    ) -> torch.Tensor:
        """
        Apply Bézier density weights to modulate feature importance.
        
        Args:
            features: [B, N, D] attended features
            bezier_density: [B, 1, H, W] density weights (0-1 range)
            seq_length: Target sequence length N
            
        Returns:
            modulated_features: [B, N, D] density-modulated features
        """
        B, _, D = features.shape
        
        # Flatten density map to sequence: [B, 1, H*W]
        density_flat = bezier_density.view(B, 1, -1)
        
        # Interpolate to match sequence length if needed
        if density_flat.size(2) != seq_length:
            density_flat = F.interpolate(
                density_flat, 
                size=seq_length, 
                mode='linear', 
                align_corners=False
            )
        
        # Process density through convolutional layer: [B, D, N]
        density_weights = self.density_proj(density_flat)  # [B, D, N]
        density_weights = self.density_gate(density_weights)  # Sigmoid activation
        
        # Transpose for element-wise multiplication: [B, N, D]
        density_weights = density_weights.transpose(1, 2)
        
        # Element-wise multiplication for density modulation
        # Higher density regions get enhanced attention
        modulated_features = features * (1.0 + density_weights)
        
        return modulated_features
    
    def _apply_adain_style_transfer(
        self,
        content_features: torch.Tensor,
        style_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply Adaptive Instance Normalization for style transfer.
        
        AdaIN transfers style statistics (mean & std) to content features
        while preserving content structure.
        
        Args:
            content_features: [B, N, D] content features (modulated)
            style_features: [B, style_dim] style reference features
            
        Returns:
            styled_features: [B, N, D] style-transferred features
        """
        B, N, D = content_features.shape
        
        # Generate style parameters (mean and std) from style features
        style_params = self.style_mlp(style_features)  # [B, D*2]
        style_mean, style_std = style_params.chunk(2, dim=1)  # Each: [B, D]
        
        # Reshape for broadcasting: [B, D, 1]
        style_mean = style_mean.unsqueeze(2)
        style_std = style_std.unsqueeze(2)
        
        # Transpose for normalization: [B, D, N]
        content_transposed = content_features.transpose(1, 2)
        
        # Instance normalization of content features
        normalized_content = self.instance_norm(content_transposed)
        
        # Apply style statistics: AdaIN(x) = σ_s * (x - μ_x) / σ_x + μ_s
        styled_features = normalized_content * (style_std + 1.0) + style_mean
        
        # Transpose back: [B, N, D]
        styled_features = styled_features.transpose(1, 2)
        
        return styled_features
    
    def _spatial_fusion_block(
        self,
        styled_features: torch.Tensor,
        original_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine styled features with original spatial features.
        
        Args:
            styled_features: [B, N, D] style-transferred features
            original_features: [B, N, D] original spatial features
            
        Returns:
            fused_features: [B, N, D] final fused output
        """
        # Concatenate styled and original features: [B, N, 2*D]
        fusion_input = torch.cat([styled_features, original_features], dim=-1)
        
        # Process through fusion layers
        fused_features = self.fusion_layers(fusion_input)  # [B, N, D]
        
        # Residual connection with original features
        fused_features = fused_features + original_features
        
        # Final layer norm
        fused_features = self.layer_norm(fused_features)
        
        return fused_features
    
    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class StyleConsistencyLoss(nn.Module):
    """
    Perceptual style loss combining content preservation and style transfer.
    """
    def __init__(self, style_weight: float = 1.0, content_weight: float = 1.0):
        super().__init__()
        self.style_weight = style_weight
        self.content_weight = content_weight
    
    def forward(
        self,
        generated: torch.Tensor,
        style_target: torch.Tensor,
        content_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute style consistency loss.
        
        Args:
            generated: [B, C, H, W] generated features
            style_target: [B, C, H, W] style reference
            content_target: [B, C, H, W] content reference
            
        Returns:
            total_loss: Combined style and content loss
        """
        # Style loss (Gram matrix comparison)
        style_loss = self._compute_style_loss(generated, style_target)
        
        # Content loss (feature similarity)
        content_loss = self._compute_content_loss(generated, content_target)
        
        total_loss = (
            self.style_weight * style_loss +
            self.content_weight * content_loss
        )
        
        return total_loss
    
    def _compute_style_loss(
        self,
        generated: torch.Tensor,
        style_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Gram matrix-based style loss."""
        def gram_matrix(features):
            B, C, H, W = features.size()
            features_flat = features.view(B, C, H * W)
            gram = torch.bmm(features_flat, features_flat.transpose(1, 2))
            return gram / (C * H * W)
        
        generated_gram = gram_matrix(generated)
        target_gram = gram_matrix(style_target)
        
        return F.mse_loss(generated_gram, target_gram)
    
    def _compute_content_loss(
        self,
        generated: torch.Tensor,
        content_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute content preservation loss."""
        return F.mse_loss(generated, content_target)


class DensityWeightedLoss(nn.Module):
    """
    Loss function that weights errors by Bézier density importance.
    """
    def __init__(self, base_weight: float = 1.0, density_weight: float = 0.5):
        super().__init__()
        self.base_weight = base_weight
        self.density_weight = density_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        density_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute density-weighted loss.
        
        Args:
            predictions: [B, C, H, W] model predictions
            targets: [B, C, H, W] ground truth targets
            density_map: [B, 1, H, W] Bézier density weights
            
        Returns:
            weighted_loss: Density-weighted MSE loss
        """
        # Base MSE loss
        base_loss = F.mse_loss(predictions, targets, reduction='none')  # [B, C, H, W]
        
        # Apply density weighting
        # Higher density regions get more importance
        density_weights = 1.0 + self.density_weight * density_map  # [B, 1, H, W]
        weighted_loss = base_loss * density_weights
        
        return weighted_loss.mean()


# Integration utilities for EasyControl pipeline
def integrate_style_bezier_fusion_with_transformer(
    transformer_block: nn.Module,
    style_bezier_fusion: StyleBezierFusionModule,
    insertion_point: str = "after_attention"
) -> nn.Module:
    """
    Integrate StyleBezierFusionModule into transformer block at optimal location.
    
    Args:
        transformer_block: Transformer block layer
        style_bezier_fusion: StyleBezierFusionModule instance
        insertion_point: Where to insert the module
        
    Returns:
        enhanced_block: Transformer block with integrated style fusion
    """
    class EnhancedTransformerBlock(nn.Module):
        def __init__(self, original_block, fusion_module):
            super().__init__()
            self.original_block = original_block
            self.fusion_module = fusion_module
        
        def forward(self, hidden_states, style_features, bezier_density, text_features=None, **kwargs):
            # Process through original transformer block
            hidden_states = self.original_block(hidden_states, **kwargs)
            
            # Apply style-Bézier fusion after attention processing
            if insertion_point == "after_attention" and style_features is not None:
                hidden_states = self.fusion_module(
                    hidden_states, style_features, bezier_density, text_features
                )
            
            return hidden_states
    
    return EnhancedTransformerBlock(transformer_block, style_bezier_fusion)


# Factory functions
def create_style_bezier_fusion(device: str = "cuda", **kwargs) -> StyleBezierFusionModule:
    """
    Factory function to create a StyleBezierFusionModule with sensible defaults.
    
    Args:
        device: Target device for the module
        **kwargs: Additional arguments for StyleBezierFusionModule
        
    Returns:
        fusion_module: Initialized StyleBezierFusionModule
    """
    fusion_module = StyleBezierFusionModule(**kwargs)
    fusion_module.to(device)
    return fusion_module


def create_complete_style_bezier_pipeline(
    feature_dim: int = 3072,
    style_dim: int = 768,
    device: str = "cuda"
) -> Tuple[StyleBezierFusionModule, StyleConsistencyLoss, DensityWeightedLoss]:
    """
    Create complete StyleBezierFusion pipeline with loss functions.
    
    Args:
        feature_dim: Feature dimension (should match transformer inner_dim)
        style_dim: Style feature dimension
        device: Target device
        
    Returns:
        fusion_module: Main fusion module
        style_loss: Style consistency loss
        density_loss: Density-weighted loss
    """
    # Create main fusion module
    fusion_module = StyleBezierFusionModule(
        feature_dim=feature_dim,
        style_dim=style_dim,
        num_heads=8
    ).to(device)
    
    # Create loss functions
    style_loss = StyleConsistencyLoss(
        style_weight=1.0,
        content_weight=0.5
    ).to(device)
    
    density_loss = DensityWeightedLoss(
        base_weight=1.0,
        density_weight=0.5
    ).to(device)
    
    return fusion_module, style_loss, density_loss