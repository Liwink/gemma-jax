"""Attention module"""

import jax
import jax.numpy as jnp

MASK_VALUE = -1e10


def scaled_dot_product_attention(
    k: jax.Array,  # (batch_size, num_heads, seq_len, head_dim)
    v: jax.Array,  # (batch_size, num_heads, seq_len, head_dim)
    q: jax.Array,  # (batch_size, num_heads, seq_len, head_dim)
    mask: jax.Array,  # (batch_size, seq_len, seq_len)
) -> jax.Array:
    """
    Compute the scaled dot-product attention.

    Args:
        q (jax.Array): Query tensor of shape (batch_size, num_heads, seq_len, head_dim).
        k (jax.Array): Key tensor of shape (batch_size, num_heads, seq_len, head_dim).
        v (jax.Array): Value tensor of shape (batch_size, num_heads, seq_len, head_dim).
        mask (jax.Array): Boolean mask tensor of shape (batch_size, seq_len, seq_len).

    Returns:
        jax.Array: The output tensor after applying attention, of shape (batch_size,
        num_heads, seq_len, head_dim).
    """

    attention_scores = jnp.einsum(
        "b h t d, b h T d -> b h t T", q, k
    )  # (batch_size, num_heads, seq_len, seq_len)

    # Normalize
    attention_scores = attention_scores / jnp.sqrt(q.shape[-1])

    # Apply mask
    attention_scores = jnp.where(mask[:, None, :, :], attention_scores, MASK_VALUE)

    # Softmax
    attention_weights = jax.nn.softmax(attention_scores, axis=-1)

    # Apply attention weights
    output = jnp.einsum("b h t T, b h T d -> b h t d", attention_weights, v)

    return output
