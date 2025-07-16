"""
Fallback attention mechanism patch for CLIP model cuDNN errors.
This module provides a robust attention fallback when cuDNN scaled_dot_product_attention fails.
"""

import torch
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Store original function globally to avoid recursion
_original_scaled_dot_product_attention = None

def safe_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Safe implementation of scaled dot product attention with fallback.

    Tries to use torch.nn.functional.scaled_dot_product_attention first,
    then falls back to manual implementation if cuDNN fails.

    Args:
        query: Query tensor [batch_size, num_heads, seq_len, head_dim]
        key: Key tensor [batch_size, num_heads, seq_len, head_dim]
        value: Value tensor [batch_size, num_heads, seq_len, head_dim]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal attention
        scale: Optional scaling factor

    Returns:
        Attention output tensor
    """
    global _original_scaled_dot_product_attention

    try:
        # Try to use the original optimized scaled_dot_product_attention
        if _original_scaled_dot_product_attention is not None:
            return _original_scaled_dot_product_attention(
                query, key, value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal
            )
        else:
            # Fallback to manual implementation if original not available
            return _manual_scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )
    except RuntimeError as e:
        if "cuDNN Frontend error" in str(e) or "No execution plans support" in str(e):
            logger.warning(f"cuDNN attention failed, falling back to manual implementation: {e}")
            return _manual_scaled_dot_product_attention(
                query, key, value, attn_mask, dropout_p, is_causal, scale
            )
        else:
            # Re-raise if it's not a cuDNN error
            raise e

def _manual_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Manual implementation of scaled dot product attention.

    This is a fallback implementation that doesn't rely on cuDNN.
    """
    batch_size, num_heads, seq_len, head_dim = query.shape

    # Calculate scale factor
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask if needed
    if is_causal:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=query.device), diagonal=1)
        scores = scores.masked_fill(causal_mask.bool(), float('-inf'))

    # Apply custom attention mask if provided
    if attn_mask is not None:
        scores = scores + attn_mask

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Apply dropout if needed
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=torch.is_grad_enabled())

    # Apply attention to values
    output = torch.matmul(attn_weights, value)

    return output

def patch_transformers_clip_attention():
    """
    Patch the transformers library CLIP attention to use safe_scaled_dot_product_attention.

    This function monkey-patches the transformers library to use our safe attention
    implementation instead of the default scaled_dot_product_attention.
    """
    global _original_scaled_dot_product_attention

    try:
        import transformers.models.clip.modeling_clip as clip_modeling

        # Store original function globally to avoid recursion
        if _original_scaled_dot_product_attention is None:
            _original_scaled_dot_product_attention = F.scaled_dot_product_attention

        # Replace with our safe implementation
        F.scaled_dot_product_attention = safe_scaled_dot_product_attention

        logger.info("Successfully patched transformers CLIP attention with fallback mechanism")

    except ImportError:
        logger.warning("Could not import transformers.models.clip.modeling_clip for patching")
    except Exception as e:
        logger.error(f"Failed to patch transformers CLIP attention: {e}")

def restore_original_attention():
    """
    Restore the original scaled_dot_product_attention function.
    """
    global _original_scaled_dot_product_attention

    try:
        if _original_scaled_dot_product_attention is not None:
            F.scaled_dot_product_attention = _original_scaled_dot_product_attention
            logger.info("Restored original scaled_dot_product_attention")

    except Exception as e:
        logger.error(f"Failed to restore original attention: {e}")