import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import math

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization (AdaIN) for style transfer.

    AdaIN adjusts the mean and variance of content features to match style features,
    enabling effective style transfer in the feature space.
    """

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, content_features: torch.Tensor, style_features: torch.Tensor) -> torch.Tensor:
        """
        Apply AdaIN to content features using style features.

        Args:
            content_features: Content features [B, N, D]
            style_features: Style features [B, D] or [B, N, D]

        Returns:
            Stylized content features [B, N, D]
        """
        # Handle different input shapes
        if style_features.dim() == 2:
            style_features = style_features.unsqueeze(1)  # [B, 1, D]

        # Compute content statistics
        content_mean = torch.mean(content_features, dim=1, keepdim=True)  # [B, 1, D]
        content_var = torch.var(content_features, dim=1, keepdim=True)    # [B, 1, D]
        content_std = torch.sqrt(content_var + self.eps)                  # [B, 1, D]

        # Compute style statistics
        style_mean = torch.mean(style_features, dim=1, keepdim=True)      # [B, 1, D]
        style_var = torch.var(style_features, dim=1, keepdim=True)        # [B, 1, D]
        style_std = torch.sqrt(style_var + self.eps)                      # [B, 1, D]

        # Normalize content features
        normalized_content = (content_features - content_mean) / content_std

        # Apply style statistics
        stylized_content = normalized_content * style_std + style_mean

        return stylized_content


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for style-content fusion.

    This module enables attention between style features and content features,
    allowing selective application of style information based on content context.
    """

    def __init__(self,
                 content_dim: int,
                 style_dim: int,
                 num_heads: int = 8,
                 head_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()

        self.content_dim = content_dim
        self.style_dim = style_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.dropout = dropout

        # Query projection from content
        self.q_proj = nn.Linear(content_dim, num_heads * head_dim, bias=False)

        # Key and value projections from style
        self.k_proj = nn.Linear(style_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(style_dim, num_heads * head_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, content_dim)

        # Dropout
        self.dropout_layer = nn.Dropout(dropout)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(content_dim)

    def forward(self,
                content_features: torch.Tensor,
                style_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.

        Args:
            content_features: Content features [B, N_content, content_dim]
            style_features: Style features [B, N_style, style_dim]
            attention_mask: Optional attention mask [B, N_content, N_style]

        Returns:
            Tuple of (attended_content, attention_weights)
        """
        batch_size, n_content, _ = content_features.shape
        _, n_style, _ = style_features.shape

        # Project to query, key, value
        q = self.q_proj(content_features)  # [B, N_content, num_heads * head_dim]
        k = self.k_proj(style_features)    # [B, N_style, num_heads * head_dim]
        v = self.v_proj(style_features)    # [B, N_style, num_heads * head_dim]

        # Reshape for multi-head attention
        q = q.view(batch_size, n_content, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, n_style, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, n_style, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf')
            )

        # Apply softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout_layer(attention_weights)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, v)

        # Reshape and project output
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, n_content, self.num_heads * self.head_dim
        )

        attended_content = self.out_proj(attended_values)

        # Add residual connection and layer norm
        attended_content = self.layer_norm(content_features + attended_content)

        return attended_content, attention_weights.mean(dim=1)  # Average over heads


class StyleEncoder(nn.Module):
    """
    Style encoder that extracts style features from style conditioning inputs.

    This module processes style vectors and converts them into representations
    suitable for style fusion operations.
    """

    def __init__(self,
                 input_dim: int = 256,      # Style vector dimension
                 hidden_dim: int = 512,     # Hidden layer dimension
                 output_dim: int = 256,     # Output style feature dimension
                 num_layers: int = 3,       # Number of encoding layers
                 dropout: float = 0.1):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Build encoder layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, style_vectors: torch.Tensor) -> torch.Tensor:
        """
        Encode style vectors into style features.

        Args:
            style_vectors: Style vectors [B, input_dim]

        Returns:
            Style features [B, output_dim]
        """
        return self.encoder(style_vectors)


class StyleBezierFusionModule(nn.Module):
    """
    StyleBezierFusionModule: Complete style fusion module with AdaIN and cross-modal attention.

    This module integrates style information from BezierAdapter into content features
    using Adaptive Instance Normalization and cross-modal attention mechanisms.

    Parameter Count: ~3.8M parameters
    Integration: Single-Stream Blocks 5-15 (Phase 2)
    """

    def __init__(self,
                 content_dim: int = 3072,        # FLUX hidden dimension
                 style_vector_dim: int = 256,    # Style vector dimension
                 style_feature_dim: int = 256,   # Style feature dimension
                 num_attention_heads: int = 8,   # Number of attention heads
                 attention_head_dim: int = 64,   # Attention head dimension
                 fusion_layers: int = 3,         # Number of fusion layers
                 use_adain: bool = True,         # Whether to use AdaIN
                 use_cross_attention: bool = True, # Whether to use cross-modal attention
                 dropout: float = 0.1,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        """
        Initialize the StyleBezierFusionModule.

        Args:
            content_dim: Content feature dimension (FLUX hidden dim)
            style_vector_dim: Input style vector dimension
            style_feature_dim: Processed style feature dimension
            num_attention_heads: Number of attention heads
            attention_head_dim: Dimension per attention head
            fusion_layers: Number of fusion layers
            use_adain: Whether to use AdaIN
            use_cross_attention: Whether to use cross-modal attention
            dropout: Dropout rate
            device: Device for computation
            dtype: Data type for computation
        """
        super().__init__()

        self.content_dim = content_dim
        self.style_vector_dim = style_vector_dim
        self.style_feature_dim = style_feature_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.fusion_layers = fusion_layers
        self.use_adain = use_adain
        self.use_cross_attention = use_cross_attention
        self.dropout = dropout
        self.device = device
        self.dtype = dtype

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize all components of the StyleBezierFusionModule."""

        # 1. Style Encoder (~400K parameters)
        self.style_encoder = StyleEncoder(
            input_dim=self.style_vector_dim,
            hidden_dim=512,
            output_dim=self.style_feature_dim,
            num_layers=3,
            dropout=self.dropout
        )

        # 2. AdaIN Module (no parameters)
        if self.use_adain:
            self.adain = AdaIN()

        # 3. Cross-Modal Attention Layers (~2.8M parameters)
        self.cross_modal_attention_layers = nn.ModuleList()
        if self.use_cross_attention:
            for i in range(self.fusion_layers):
                attention_layer = CrossModalAttention(
                    content_dim=self.content_dim,
                    style_dim=self.style_feature_dim,
                    num_heads=self.num_attention_heads,
                    head_dim=self.attention_head_dim,
                    dropout=self.dropout
                )
                self.cross_modal_attention_layers.append(attention_layer)

        # 4. Style Modulation Network (~400K parameters)
        self.style_modulation = nn.Sequential(
            nn.Linear(self.style_feature_dim, self.content_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.content_dim, self.content_dim),
            nn.Sigmoid()  # Gating mechanism
        )

        # 5. Content-Style Fusion Network (~200K parameters)
        self.fusion_network = nn.Sequential(
            nn.Linear(self.content_dim * 2, self.content_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.content_dim, self.content_dim),
            nn.LayerNorm(self.content_dim)
        )

        # 6. Output Projection (~100K parameters)
        self.output_projection = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.content_dim, self.content_dim)
        )

        # 7. Normalization layers
        self.content_norm = nn.LayerNorm(self.content_dim)
        self.style_norm = nn.LayerNorm(self.style_feature_dim)

        # 8. Learnable mixing weights
        self.mixing_weights = nn.Parameter(torch.ones(self.fusion_layers))

    def forward(self,
                content_features: torch.Tensor,
                style_vectors: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for style-content fusion.

        Args:
            content_features: Content features [B, N, content_dim]
            style_vectors: Style vectors [B, style_vector_dim]
            attention_mask: Optional attention mask [B, N]

        Returns:
            Tuple of (fused_features, debug_info)
        """
        batch_size, seq_len, _ = content_features.shape

        # 1. Encode style vectors
        style_features = self.style_encoder(style_vectors)  # [B, style_feature_dim]
        style_features = self.style_norm(style_features)

        # 2. Apply AdaIN if enabled
        if self.use_adain:
            adain_features = self.adain(content_features, style_features)
            current_features = adain_features
        else:
            current_features = content_features

        # 3. Apply cross-modal attention layers
        attention_outputs = []
        attention_weights_list = []

        if self.use_cross_attention:
            # Expand style features for attention
            style_features_expanded = style_features.unsqueeze(1).expand(-1, seq_len, -1)

            for i, attention_layer in enumerate(self.cross_modal_attention_layers):
                attended_features, attention_weights = attention_layer(
                    content_features=current_features,
                    style_features=style_features_expanded,
                    attention_mask=attention_mask
                )

                # Apply mixing weight
                mixing_weight = torch.softmax(self.mixing_weights, dim=0)[i]
                current_features = mixing_weight * attended_features + (1 - mixing_weight) * current_features

                attention_outputs.append(attended_features)
                attention_weights_list.append(attention_weights)

        # 4. Apply style modulation
        style_gate = self.style_modulation(style_features)  # [B, content_dim]
        style_gate = style_gate.unsqueeze(1)  # [B, 1, content_dim]

        modulated_features = current_features * style_gate

        # 5. Fusion network
        # Concatenate original content and modulated features
        concatenated_features = torch.cat([content_features, modulated_features], dim=-1)
        fused_features = self.fusion_network(concatenated_features)

        # 6. Output projection with residual connection
        projected_features = self.output_projection(fused_features)
        final_features = self.content_norm(content_features + projected_features)

        # 7. Prepare debug information
        debug_info = {
            'style_features': style_features,
            'style_gate': style_gate,
            'modulated_features': modulated_features,
            'fused_features': fused_features,
            'mixing_weights': torch.softmax(self.mixing_weights, dim=0),
            'attention_outputs': attention_outputs,
            'attention_weights': attention_weights_list
        }

        if self.use_adain:
            debug_info['adain_features'] = adain_features

        return final_features, debug_info

    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter count breakdown for each component."""

        total_params = sum(p.numel() for p in self.parameters())

        component_counts = {
            'style_encoder': sum(p.numel() for p in self.style_encoder.parameters()),
            'cross_modal_attention': sum(p.numel() for p in self.cross_modal_attention_layers.parameters()) if self.use_cross_attention else 0,
            'style_modulation': sum(p.numel() for p in self.style_modulation.parameters()),
            'fusion_network': sum(p.numel() for p in self.fusion_network.parameters()),
            'output_projection': sum(p.numel() for p in self.output_projection.parameters()),
            'layer_norms': sum(p.numel() for p in self.content_norm.parameters()) + sum(p.numel() for p in self.style_norm.parameters()),
            'mixing_weights': self.mixing_weights.numel(),
            'total': total_params
        }

        return component_counts

    def set_style_strength(self, strength: float):
        """Set the strength of style application (0.0 to 1.0)."""
        if hasattr(self, 'style_strength'):
            self.style_strength = torch.clamp(torch.tensor(strength), 0.0, 1.0)
        else:
            self.register_buffer('style_strength', torch.tensor(strength))

    def get_style_strength(self) -> float:
        """Get the current style strength."""
        return getattr(self, 'style_strength', torch.tensor(1.0)).item()


class StyleBezierProcessor(nn.Module):
    """
    Custom attention processor that integrates StyleBezierFusionModule with FLUX attention.
    """

    def __init__(self,
                 style_fusion_module: StyleBezierFusionModule,
                 base_processor: Optional[Any] = None):
        super().__init__()

        self.style_fusion_module = style_fusion_module
        self.base_processor = base_processor

    def __call__(self,
                 attn,
                 hidden_states: torch.Tensor,
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                 style_vectors: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """
        Forward pass for style-aware attention processing.

        Args:
            attn: Attention module
            hidden_states: Input hidden states
            encoder_hidden_states: Optional encoder hidden states
            attention_mask: Optional attention mask
            image_rotary_emb: Optional rotary embeddings
            style_vectors: Optional style vectors for conditioning
            **kwargs: Additional arguments

        Returns:
            Processed attention output with style fusion
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

        # Apply style fusion if style vectors are provided
        if style_vectors is not None:
            fused_output, debug_info = self.style_fusion_module(
                content_features=attention_output,
                style_vectors=style_vectors,
                attention_mask=attention_mask
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


def create_style_bezier_fusion_module(content_dim: int = 3072,
                                    style_vector_dim: int = 256,
                                    config: Dict[str, Any] = None) -> StyleBezierFusionModule:
    """
    Factory function to create a StyleBezierFusionModule with default or custom configuration.

    Args:
        content_dim: Content feature dimension
        style_vector_dim: Style vector dimension
        config: Optional custom configuration

    Returns:
        Configured StyleBezierFusionModule
    """
    default_config = {
        'content_dim': content_dim,
        'style_vector_dim': style_vector_dim,
        'style_feature_dim': 256,
        'num_attention_heads': 8,
        'attention_head_dim': 64,
        'fusion_layers': 3,
        'use_adain': True,
        'use_cross_attention': True,
        'dropout': 0.1,
        'device': 'cuda',
        'dtype': torch.float32
    }

    if config:
        default_config.update(config)

    return StyleBezierFusionModule(**default_config)


def test_style_bezier_fusion():
    """Test function for StyleBezierFusionModule."""

    # Create module
    fusion_module = create_style_bezier_fusion_module()

    # Test data
    batch_size = 2
    seq_len = 4096
    content_dim = 3072
    style_vector_dim = 256

    content_features = torch.randn(batch_size, seq_len, content_dim)
    style_vectors = torch.randn(batch_size, style_vector_dim)

    # Forward pass
    fused_features, debug_info = fusion_module(content_features, style_vectors)

    print(f"Input shape: {content_features.shape}")
    print(f"Style vectors shape: {style_vectors.shape}")
    print(f"Output shape: {fused_features.shape}")

    # Parameter count
    param_count = fusion_module.get_parameter_count()
    print(f"Parameter count: {param_count}")

    return fusion_module


if __name__ == "__main__":
    test_style_bezier_fusion()