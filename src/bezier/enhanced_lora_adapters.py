"""
Enhanced LoRA Adapter System for BezierAdapter Multi-Modal Conditioning

This module extends the existing EasyControl LoRA system to support:
- Style Branch LoRA (r=64): Style vector conditioning
- Text Branch LoRA (r=64): Enhanced text conditioning
- Density Branch LoRA (r=64): Bézier density conditioning
- Backward compatibility with existing spatial/subject conditioning

Total additional parameters: ~3.6M (3 branches × 1.2M each)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union, Tuple
from enum import Enum
import math

class ConditionType(Enum):
    """Enumeration of condition types for LoRA branches."""
    SPATIAL = "spatial"
    SUBJECT = "subject"
    STYLE = "style"
    TEXT = "text"
    DENSITY = "density"

class EnhancedLoRALinearLayer(nn.Module):
    """
    Enhanced LoRA Linear Layer with support for different condition types.

    Extends the original LoRALinearLayer to support:
    - Branch-specific conditioning types
    - Adaptive masking for different modalities
    - Feature-aware conditioning
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 rank: int = 64,
                 condition_type: ConditionType = ConditionType.SPATIAL,
                 network_alpha: Optional[float] = None,
                 device: Optional[Union[torch.device, str]] = None,
                 dtype: Optional[torch.dtype] = None,
                 cond_width: int = 512,
                 cond_height: int = 512,
                 branch_id: int = 0,
                 total_branches: int = 1,
                 use_feature_conditioning: bool = False,
                 feature_dim: int = 256):
        """
        Initialize Enhanced LoRA Linear Layer.

        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            rank: LoRA rank (default: 64 for BezierAdapter branches)
            condition_type: Type of conditioning (spatial, subject, style, text, density)
            network_alpha: Network alpha for LoRA scaling
            device: Device to place parameters
            dtype: Data type for parameters
            cond_width: Condition width for spatial calculations
            cond_height: Condition height for spatial calculations
            branch_id: ID of this branch (for masking)
            total_branches: Total number of branches
            use_feature_conditioning: Whether to use feature-based conditioning
            feature_dim: Dimension of conditioning features
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.condition_type = condition_type
        self.network_alpha = network_alpha
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.branch_id = branch_id
        self.total_branches = total_branches
        self.use_feature_conditioning = use_feature_conditioning
        self.feature_dim = feature_dim

        # Core LoRA layers
        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)

        # Initialize weights
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

        # Condition-specific enhancements
        if condition_type == ConditionType.STYLE:
            # Style conditioning: feature-based modulation
            self.style_modulator = nn.Sequential(
                nn.Linear(feature_dim, rank),
                nn.ReLU(),
                nn.Linear(rank, rank),
                nn.Sigmoid()
            )
        elif condition_type == ConditionType.TEXT:
            # Text conditioning: attention-based enhancement
            self.text_attention = nn.MultiheadAttention(
                embed_dim=rank,
                num_heads=8,
                batch_first=True,
                device=device,
                dtype=dtype
            )
        elif condition_type == ConditionType.DENSITY:
            # Density conditioning: spatial-aware processing
            self.density_processor = nn.Sequential(
                nn.Linear(rank, rank * 2),
                nn.ReLU(),
                nn.Linear(rank * 2, rank),
                nn.Tanh()
            )

        # Feature conditioning network (if enabled)
        if use_feature_conditioning:
            self.feature_conditioner = nn.Sequential(
                nn.Linear(feature_dim, rank),
                nn.ReLU(),
                nn.Linear(rank, rank),
                nn.Sigmoid()
            )

    def get_condition_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Generate condition-specific mask for different branch types.

        Args:
            hidden_states: Input hidden states [B, seq_len, dim]

        Returns:
            Mask tensor for this branch
        """
        batch_size, seq_len, dim = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Calculate condition sizes based on type
        if self.condition_type in [ConditionType.SPATIAL, ConditionType.SUBJECT]:
            # Original spatial/subject masking
            cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
            block_size = seq_len - cond_size * self.total_branches

            # Create mask for this specific branch
            mask = torch.ones((batch_size, seq_len, dim), device=device, dtype=dtype)
            mask[:, :block_size + self.branch_id * cond_size, :] = 0
            mask[:, block_size + (self.branch_id + 1) * cond_size:, :] = 0

        elif self.condition_type == ConditionType.STYLE:
            # Style conditioning: global feature modulation (no spatial masking)
            mask = torch.ones((batch_size, seq_len, dim), device=device, dtype=dtype)

        elif self.condition_type == ConditionType.TEXT:
            # Text conditioning: enhanced text token processing
            # Assume text tokens are at the beginning
            text_seq_len = seq_len // 4  # Approximate text length
            mask = torch.zeros((batch_size, seq_len, dim), device=device, dtype=dtype)
            mask[:, :text_seq_len, :] = 1

        elif self.condition_type == ConditionType.DENSITY:
            # Density conditioning: spatial density-aware masking
            # Apply to spatial regions with density information
            cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
            block_size = seq_len - cond_size * self.total_branches

            mask = torch.ones((batch_size, seq_len, dim), device=device, dtype=dtype)
            # Focus on spatial regions for density conditioning
            mask[:, :block_size, :] = 0.5  # Reduced influence on main content
            mask[:, block_size + self.branch_id * cond_size:block_size + (self.branch_id + 1) * cond_size, :] = 1

        else:
            # Default: no masking
            mask = torch.ones((batch_size, seq_len, dim), device=device, dtype=dtype)

        return mask

    def forward(self,
                hidden_states: torch.Tensor,
                condition_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with enhanced conditioning.

        Args:
            hidden_states: Input hidden states [B, seq_len, dim]
            condition_features: Optional conditioning features [B, feature_dim]

        Returns:
            LoRA output tensor
        """
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        # Apply condition-specific masking
        mask = self.get_condition_mask(hidden_states)
        masked_states = mask * hidden_states

        # Core LoRA computation
        down_states = self.down(masked_states.to(dtype))

        # Apply condition-specific enhancements
        if self.condition_type == ConditionType.STYLE and condition_features is not None:
            # Style modulation
            style_weights = self.style_modulator(condition_features).unsqueeze(1)  # [B, 1, rank]
            down_states = down_states * style_weights

        elif self.condition_type == ConditionType.TEXT:
            # Text attention enhancement
            batch_size, seq_len, rank = down_states.shape
            down_states_reshaped = down_states.view(batch_size * seq_len, 1, rank)
            enhanced_states, _ = self.text_attention(
                down_states_reshaped, down_states_reshaped, down_states_reshaped
            )
            down_states = enhanced_states.view(batch_size, seq_len, rank)

        elif self.condition_type == ConditionType.DENSITY:
            # Density processing
            down_states = self.density_processor(down_states)

        # Apply feature conditioning if enabled
        if self.use_feature_conditioning and condition_features is not None:
            feature_weights = self.feature_conditioner(condition_features).unsqueeze(1)  # [B, 1, rank]
            down_states = down_states * feature_weights

        # Final projection
        up_states = self.up(down_states)

        # Apply network alpha scaling
        if self.network_alpha is not None:
            up_states *= self.network_alpha / self.rank

        return up_states.to(orig_dtype)

class BezierLoRABranch(nn.Module):
    """
    Specialized LoRA branch for BezierAdapter conditioning.

    This module encapsulates a complete LoRA branch with:
    - Query, Key, Value projections
    - Optional output projection
    - Condition-specific processing
    """

    def __init__(self,
                 dim: int,
                 condition_type: ConditionType,
                 rank: int = 64,
                 network_alpha: Optional[float] = None,
                 device: Optional[Union[torch.device, str]] = None,
                 dtype: Optional[torch.dtype] = None,
                 cond_width: int = 512,
                 cond_height: int = 512,
                 branch_id: int = 0,
                 total_branches: int = 1,
                 use_output_projection: bool = False,
                 feature_dim: int = 256):
        """
        Initialize BezierLoRA Branch.

        Args:
            dim: Feature dimension
            condition_type: Type of conditioning
            rank: LoRA rank
            network_alpha: Network alpha for scaling
            device: Device to place parameters
            dtype: Data type
            cond_width: Condition width
            cond_height: Condition height
            branch_id: Branch identifier
            total_branches: Total number of branches
            use_output_projection: Whether to include output projection
            feature_dim: Feature dimension for conditioning
        """
        super().__init__()

        self.condition_type = condition_type
        self.rank = rank
        self.dim = dim
        self.use_output_projection = use_output_projection

        # Query, Key, Value LoRA layers
        self.q_lora = EnhancedLoRALinearLayer(
            dim, dim, rank, condition_type, network_alpha, device, dtype,
            cond_width, cond_height, branch_id, total_branches,
            use_feature_conditioning=True, feature_dim=feature_dim
        )

        self.k_lora = EnhancedLoRALinearLayer(
            dim, dim, rank, condition_type, network_alpha, device, dtype,
            cond_width, cond_height, branch_id, total_branches,
            use_feature_conditioning=True, feature_dim=feature_dim
        )

        self.v_lora = EnhancedLoRALinearLayer(
            dim, dim, rank, condition_type, network_alpha, device, dtype,
            cond_width, cond_height, branch_id, total_branches,
            use_feature_conditioning=True, feature_dim=feature_dim
        )

        # Optional output projection
        if use_output_projection:
            self.proj_lora = EnhancedLoRALinearLayer(
                dim, dim, rank, condition_type, network_alpha, device, dtype,
                cond_width, cond_height, branch_id, total_branches,
                use_feature_conditioning=True, feature_dim=feature_dim
            )

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                hidden_states: torch.Tensor,
                condition_features: Optional[torch.Tensor] = None,
                lora_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for LoRA branch.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            hidden_states: Hidden states for masking
            condition_features: Conditioning features
            lora_weight: LoRA weight scaling

        Returns:
            Tuple of (enhanced_query, enhanced_key, enhanced_value)
        """
        # Apply LoRA to Q, K, V
        q_lora_out = self.q_lora(hidden_states, condition_features)
        k_lora_out = self.k_lora(hidden_states, condition_features)
        v_lora_out = self.v_lora(hidden_states, condition_features)

        # Add to original projections
        enhanced_query = query + lora_weight * q_lora_out
        enhanced_key = key + lora_weight * k_lora_out
        enhanced_value = value + lora_weight * v_lora_out

        return enhanced_query, enhanced_key, enhanced_value

    def forward_output_projection(self,
                                 hidden_states: torch.Tensor,
                                 condition_features: Optional[torch.Tensor] = None,
                                 lora_weight: float = 1.0) -> torch.Tensor:
        """
        Forward pass for output projection.

        Args:
            hidden_states: Hidden states
            condition_features: Conditioning features
            lora_weight: LoRA weight scaling

        Returns:
            Output projection enhancement
        """
        if self.use_output_projection:
            return lora_weight * self.proj_lora(hidden_states, condition_features)
        else:
            return torch.zeros_like(hidden_states)

class EnhancedMultiSingleStreamBlockLoraProcessor(nn.Module):
    """
    Enhanced Multi-Stream LoRA Processor for Single-Stream Blocks.

    Extends the original MultiSingleStreamBlockLoraProcessor to support:
    - BezierAdapter conditioning branches (Style, Text, Density)
    - Backward compatibility with existing spatial/subject conditioning
    - Feature-aware conditioning
    """

    def __init__(self,
                 dim: int,
                 # Legacy parameters for backward compatibility
                 ranks: List[int] = None,
                 lora_weights: List[float] = None,
                 network_alphas: List[float] = None,
                 device: Optional[Union[torch.device, str]] = None,
                 dtype: Optional[torch.dtype] = None,
                 cond_width: int = 512,
                 cond_height: int = 512,
                 n_loras: int = 1,
                 # New BezierAdapter parameters
                 bezier_condition_types: List[ConditionType] = None,
                 bezier_ranks: List[int] = None,
                 bezier_weights: List[float] = None,
                 bezier_network_alphas: List[float] = None,
                 enable_bezier_conditioning: bool = True,
                 feature_dim: int = 256):
        """
        Initialize Enhanced Multi-Stream LoRA Processor.

        Args:
            dim: Feature dimension
            ranks: Legacy LoRA ranks
            lora_weights: Legacy LoRA weights
            network_alphas: Legacy network alphas
            device: Device
            dtype: Data type
            cond_width: Condition width
            cond_height: Condition height
            n_loras: Number of legacy LoRA streams
            bezier_condition_types: BezierAdapter condition types
            bezier_ranks: BezierAdapter LoRA ranks
            bezier_weights: BezierAdapter LoRA weights
            bezier_network_alphas: BezierAdapter network alphas
            enable_bezier_conditioning: Whether to enable BezierAdapter conditioning
            feature_dim: Feature dimension for conditioning
        """
        super().__init__()

        self.dim = dim
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.n_loras = n_loras
        self.enable_bezier_conditioning = enable_bezier_conditioning
        self.feature_dim = feature_dim

        # Legacy LoRA streams (for backward compatibility)
        if ranks is None:
            ranks = [4] * n_loras
        if lora_weights is None:
            lora_weights = [1.0] * n_loras
        if network_alphas is None:
            network_alphas = [None] * n_loras

        self.lora_weights = lora_weights

        # Create legacy LoRA branches
        self.legacy_branches = nn.ModuleList()
        for i in range(n_loras):
            branch = BezierLoRABranch(
                dim=dim,
                condition_type=ConditionType.SPATIAL,  # Default to spatial for legacy
                rank=ranks[i],
                network_alpha=network_alphas[i],
                device=device,
                dtype=dtype,
                cond_width=cond_width,
                cond_height=cond_height,
                branch_id=i,
                total_branches=n_loras,
                use_output_projection=False,
                feature_dim=feature_dim
            )
            self.legacy_branches.append(branch)

        # BezierAdapter conditioning branches
        if enable_bezier_conditioning:
            if bezier_condition_types is None:
                bezier_condition_types = [ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY]
            if bezier_ranks is None:
                bezier_ranks = [64] * len(bezier_condition_types)
            if bezier_weights is None:
                bezier_weights = [1.0] * len(bezier_condition_types)
            if bezier_network_alphas is None:
                bezier_network_alphas = [None] * len(bezier_condition_types)

            self.bezier_condition_types = bezier_condition_types
            self.bezier_weights = bezier_weights

            # Create BezierAdapter LoRA branches
            self.bezier_branches = nn.ModuleList()
            for i, condition_type in enumerate(bezier_condition_types):
                branch = BezierLoRABranch(
                    dim=dim,
                    condition_type=condition_type,
                    rank=bezier_ranks[i],
                    network_alpha=bezier_network_alphas[i],
                    device=device,
                    dtype=dtype,
                    cond_width=cond_width,
                    cond_height=cond_height,
                    branch_id=i + n_loras,  # Offset by legacy branches
                    total_branches=n_loras + len(bezier_condition_types),
                    use_output_projection=False,
                    feature_dim=feature_dim
                )
                self.bezier_branches.append(branch)
        else:
            self.bezier_branches = nn.ModuleList()
            self.bezier_weights = []

    def __call__(self,
                 attn,
                 hidden_states: torch.Tensor,
                 encoder_hidden_states: Optional[torch.Tensor] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 image_rotary_emb: Optional[torch.Tensor] = None,
                 use_cond: bool = False,
                 # BezierAdapter conditioning features
                 style_features: Optional[torch.Tensor] = None,
                 text_features: Optional[torch.Tensor] = None,
                 density_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with enhanced conditioning.

        Args:
            attn: Attention module
            hidden_states: Hidden states
            encoder_hidden_states: Encoder hidden states
            attention_mask: Attention mask
            image_rotary_emb: Rotary embeddings
            use_cond: Whether to use conditioning
            style_features: Style conditioning features
            text_features: Text conditioning features
            density_features: Density conditioning features

        Returns:
            Enhanced attention output
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Original attention projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Apply legacy LoRA branches (backward compatibility)
        for i, branch in enumerate(self.legacy_branches):
            query, key, value = branch(
                query, key, value, hidden_states,
                condition_features=None,
                lora_weight=self.lora_weights[i]
            )

        # Apply BezierAdapter LoRA branches
        if self.enable_bezier_conditioning:
            for i, (branch, condition_type) in enumerate(zip(self.bezier_branches, self.bezier_condition_types)):
                # Select appropriate conditioning features
                if condition_type == ConditionType.STYLE:
                    condition_features = style_features
                elif condition_type == ConditionType.TEXT:
                    condition_features = text_features
                elif condition_type == ConditionType.DENSITY:
                    condition_features = density_features
                else:
                    condition_features = None

                query, key, value = branch(
                    query, key, value, hidden_states,
                    condition_features=condition_features,
                    lora_weight=self.bezier_weights[i]
                )

        # Reshape for attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Create attention mask for multi-stream processing
        attention_mask = self._create_multi_stream_mask(
            seq_len, hidden_states.device, hidden_states.dtype
        )

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0.0,
            is_causal=False,
            attn_mask=attention_mask
        )

        # Reshape back
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split main and condition streams
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        total_branches = self.n_loras + len(self.bezier_branches)
        block_size = seq_len - cond_size * total_branches

        cond_hidden_states = hidden_states[:, block_size:, :]
        hidden_states = hidden_states[:, :block_size, :]

        return hidden_states if not use_cond else (hidden_states, cond_hidden_states)

    def _create_multi_stream_mask(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create attention mask for multi-stream processing."""
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        total_branches = self.n_loras + len(self.bezier_branches)
        block_size = seq_len - cond_size * total_branches

        # Create mask
        mask = torch.ones((seq_len, seq_len), device=device, dtype=dtype)
        mask[:block_size, :] = 0  # Main block can attend to everything

        # Each conditioning branch can only attend to itself
        for i in range(total_branches):
            start = i * cond_size + block_size
            end = (i + 1) * cond_size + block_size
            mask[start:end, start:end] = 0

        # Apply large negative value to masked positions
        mask = mask * -1e20

        return mask.to(dtype)

class EnhancedMultiDoubleStreamBlockLoraProcessor(nn.Module):
    """
    Enhanced Multi-Stream LoRA Processor for Double-Stream Blocks.

    Extends the original MultiDoubleStreamBlockLoraProcessor to support BezierAdapter conditioning.
    """

    def __init__(self,
                 dim: int,
                 ranks: List[int] = None,
                 lora_weights: List[float] = None,
                 network_alphas: List[float] = None,
                 device: Optional[Union[torch.device, str]] = None,
                 dtype: Optional[torch.dtype] = None,
                 cond_width: int = 512,
                 cond_height: int = 512,
                 n_loras: int = 1,
                 bezier_condition_types: List[ConditionType] = None,
                 bezier_ranks: List[int] = None,
                 bezier_weights: List[float] = None,
                 bezier_network_alphas: List[float] = None,
                 enable_bezier_conditioning: bool = True,
                 feature_dim: int = 256):
        """Initialize Enhanced Multi-Stream LoRA Processor for Double-Stream Blocks."""
        super().__init__()

        self.dim = dim
        self.cond_width = cond_width
        self.cond_height = cond_height
        self.n_loras = n_loras
        self.enable_bezier_conditioning = enable_bezier_conditioning
        self.feature_dim = feature_dim

        # Initialize similar to single-stream but with output projection
        if ranks is None:
            ranks = [4] * n_loras
        if lora_weights is None:
            lora_weights = [1.0] * n_loras
        if network_alphas is None:
            network_alphas = [None] * n_loras

        self.lora_weights = lora_weights

        # Create legacy LoRA branches with output projection
        self.legacy_branches = nn.ModuleList()
        for i in range(n_loras):
            branch = BezierLoRABranch(
                dim=dim,
                condition_type=ConditionType.SPATIAL,
                rank=ranks[i],
                network_alpha=network_alphas[i],
                device=device,
                dtype=dtype,
                cond_width=cond_width,
                cond_height=cond_height,
                branch_id=i,
                total_branches=n_loras,
                use_output_projection=True,  # Double-stream uses output projection
                feature_dim=feature_dim
            )
            self.legacy_branches.append(branch)

        # BezierAdapter conditioning branches
        if enable_bezier_conditioning:
            if bezier_condition_types is None:
                bezier_condition_types = [ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY]
            if bezier_ranks is None:
                bezier_ranks = [64] * len(bezier_condition_types)
            if bezier_weights is None:
                bezier_weights = [1.0] * len(bezier_condition_types)
            if bezier_network_alphas is None:
                bezier_network_alphas = [None] * len(bezier_condition_types)

            self.bezier_condition_types = bezier_condition_types
            self.bezier_weights = bezier_weights

            self.bezier_branches = nn.ModuleList()
            for i, condition_type in enumerate(bezier_condition_types):
                branch = BezierLoRABranch(
                    dim=dim,
                    condition_type=condition_type,
                    rank=bezier_ranks[i],
                    network_alpha=bezier_network_alphas[i],
                    device=device,
                    dtype=dtype,
                    cond_width=cond_width,
                    cond_height=cond_height,
                    branch_id=i + n_loras,
                    total_branches=n_loras + len(bezier_condition_types),
                    use_output_projection=True,
                    feature_dim=feature_dim
                )
                self.bezier_branches.append(branch)
        else:
            self.bezier_branches = nn.ModuleList()
            self.bezier_weights = []

    def __call__(self,
                 attn,
                 hidden_states: torch.Tensor,
                 encoder_hidden_states: torch.Tensor,
                 attention_mask: Optional[torch.Tensor] = None,
                 image_rotary_emb: Optional[torch.Tensor] = None,
                 use_cond: bool = False,
                 style_features: Optional[torch.Tensor] = None,
                 text_features: Optional[torch.Tensor] = None,
                 density_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for double-stream attention with BezierAdapter conditioning."""

        batch_size, _, _ = hidden_states.shape

        # Context projections (encoder)
        inner_dim = 3072
        head_dim = inner_dim // attn.heads

        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        # Apply normalization to context projections
        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

        # Main hidden states projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Apply legacy LoRA branches
        for i, branch in enumerate(self.legacy_branches):
            query, key, value = branch(
                query, key, value, hidden_states,
                condition_features=None,
                lora_weight=self.lora_weights[i]
            )

        # Apply BezierAdapter LoRA branches
        if self.enable_bezier_conditioning:
            for i, (branch, condition_type) in enumerate(zip(self.bezier_branches, self.bezier_condition_types)):
                if condition_type == ConditionType.STYLE:
                    condition_features = style_features
                elif condition_type == ConditionType.TEXT:
                    condition_features = text_features
                elif condition_type == ConditionType.DENSITY:
                    condition_features = density_features
                else:
                    condition_features = None

                query, key, value = branch(
                    query, key, value, hidden_states,
                    condition_features=condition_features,
                    lora_weight=self.bezier_weights[i]
                )

        # Reshape for attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Concatenate context and main streams
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # Apply rotary embeddings
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Create attention mask
        attention_mask = self._create_double_stream_mask(
            hidden_states.shape[1], encoder_hidden_states.shape[1],
            hidden_states.device, hidden_states.dtype
        )

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            dropout_p=0.0,
            is_causal=False,
            attn_mask=attention_mask
        )

        # Reshape and split streams
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, :encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1]:],
        )

        # Output projection with LoRA
        hidden_states = attn.to_out[0](hidden_states)

        # Apply legacy LoRA output projections
        for i, branch in enumerate(self.legacy_branches):
            hidden_states = hidden_states + branch.forward_output_projection(
                hidden_states, condition_features=None, lora_weight=self.lora_weights[i]
            )

        # Apply BezierAdapter LoRA output projections
        if self.enable_bezier_conditioning:
            for i, (branch, condition_type) in enumerate(zip(self.bezier_branches, self.bezier_condition_types)):
                if condition_type == ConditionType.STYLE:
                    condition_features = style_features
                elif condition_type == ConditionType.TEXT:
                    condition_features = text_features
                elif condition_type == ConditionType.DENSITY:
                    condition_features = density_features
                else:
                    condition_features = None

                hidden_states = hidden_states + branch.forward_output_projection(
                    hidden_states, condition_features=condition_features,
                    lora_weight=self.bezier_weights[i]
                )

        # Dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # Split condition streams
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        total_branches = self.n_loras + len(self.bezier_branches)
        block_size = hidden_states.shape[1] - cond_size * total_branches

        cond_hidden_states = hidden_states[:, block_size:, :]
        hidden_states = hidden_states[:, :block_size, :]

        return (hidden_states, encoder_hidden_states, cond_hidden_states) if use_cond else (encoder_hidden_states, hidden_states)

    def _create_double_stream_mask(self, hidden_seq_len: int, encoder_seq_len: int,
                                  device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create attention mask for double-stream processing."""
        cond_size = self.cond_width // 8 * self.cond_height // 8 * 16 // 64
        total_branches = self.n_loras + len(self.bezier_branches)
        block_size = hidden_seq_len - cond_size * total_branches

        total_seq_len = encoder_seq_len + hidden_seq_len

        # Create mask
        mask = torch.ones((total_seq_len, total_seq_len), device=device, dtype=dtype)

        # Encoder tokens can attend to everything
        mask[:encoder_seq_len, :] = 0

        # Main hidden states can attend to everything
        mask[encoder_seq_len:encoder_seq_len + block_size, :] = 0

        # Each conditioning branch can only attend to itself and encoder
        for i in range(total_branches):
            start = encoder_seq_len + block_size + i * cond_size
            end = encoder_seq_len + block_size + (i + 1) * cond_size
            mask[start:end, :encoder_seq_len] = 0  # Can attend to encoder
            mask[start:end, start:end] = 0  # Can attend to itself

        # Apply large negative value to masked positions
        mask = mask * -1e20

        return mask.to(dtype)

def count_bezier_lora_parameters(processor) -> Dict[str, int]:
    """
    Count parameters in BezierAdapter LoRA processor.

    Args:
        processor: Enhanced LoRA processor

    Returns:
        Dictionary with parameter counts
    """
    counts = {
        'legacy_branches': 0,
        'bezier_branches': 0,
        'style_branch': 0,
        'text_branch': 0,
        'density_branch': 0,
        'total': 0
    }

    # Count legacy branch parameters
    for branch in processor.legacy_branches:
        counts['legacy_branches'] += sum(p.numel() for p in branch.parameters())

    # Count BezierAdapter branch parameters
    for branch, condition_type in zip(processor.bezier_branches, processor.bezier_condition_types):
        branch_params = sum(p.numel() for p in branch.parameters())
        counts['bezier_branches'] += branch_params

        if condition_type == ConditionType.STYLE:
            counts['style_branch'] += branch_params
        elif condition_type == ConditionType.TEXT:
            counts['text_branch'] += branch_params
        elif condition_type == ConditionType.DENSITY:
            counts['density_branch'] += branch_params

    counts['total'] = counts['legacy_branches'] + counts['bezier_branches']

    return counts

def create_enhanced_lora_processor(
    processor_type: str,
    dim: int,
    enable_bezier_conditioning: bool = True,
    bezier_condition_types: List[ConditionType] = None,
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs
):
    """
    Factory function to create enhanced LoRA processors.

    Args:
        processor_type: Type of processor ('single' or 'double')
        dim: Feature dimension
        enable_bezier_conditioning: Whether to enable BezierAdapter conditioning
        bezier_condition_types: List of condition types to enable
        device: Device to place parameters
        dtype: Data type
        **kwargs: Additional arguments

    Returns:
        Enhanced LoRA processor
    """
    if bezier_condition_types is None:
        bezier_condition_types = [ConditionType.STYLE, ConditionType.TEXT, ConditionType.DENSITY]

    if processor_type == 'single':
        return EnhancedMultiSingleStreamBlockLoraProcessor(
            dim=dim,
            enable_bezier_conditioning=enable_bezier_conditioning,
            bezier_condition_types=bezier_condition_types,
            device=device,
            dtype=dtype,
            **kwargs
        )
    elif processor_type == 'double':
        return EnhancedMultiDoubleStreamBlockLoraProcessor(
            dim=dim,
            enable_bezier_conditioning=enable_bezier_conditioning,
            bezier_condition_types=bezier_condition_types,
            device=device,
            dtype=dtype,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")


# Example usage
if __name__ == "__main__":
    # Create enhanced single-stream processor
    processor = create_enhanced_lora_processor(
        processor_type='single',
        dim=3072,
        enable_bezier_conditioning=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Count parameters
    param_counts = count_bezier_lora_parameters(processor)
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")

    print(f"\nTotal BezierAdapter parameters: {param_counts['bezier_branches']:,}")
    print(f"Target was ~3.6M parameters: {'✓' if param_counts['bezier_branches'] < 4e6 else '✗'}")