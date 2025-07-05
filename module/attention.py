"""Attention module"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import repeat
from .rope_simple import apply_rope
from .rms import RMSNorm

MASK_VALUE = -2.3819763e38


def scaled_dot_product_attention(
    k: jax.Array,  # (batch_size, num_key_value_heads, seq_len, head_dim)
    v: jax.Array,  # (batch_size, num_key_value_heads, seq_len, head_dim)
    q: jax.Array,  # (batch_size, num_query_heads, seq_len, head_dim)
    mask: jax.Array,  # (batch_size, seq_len, seq_len)
) -> jax.Array:  # (batch_size, num_query_heads, seq_len, head_dim)
    """
    Compute the scaled dot-product attention.

    Args:
        q (jax.Array): Query tensor of shape (batch_size, num_query_heads, seq_len, head_dim).
        k (jax.Array): Key tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim).
        v (jax.Array): Value tensor of shape (batch_size, num_key_value_heads, seq_len, head_dim).
        mask (jax.Array): Boolean mask tensor of shape (batch_size, seq_len, seq_len).

    Returns:
        jax.Array: The output tensor after applying attention, of shape (batch_size,
        num_query_heads, seq_len, head_dim).
    """
    num_key_value_heads = k.shape[1]
    num_query_heads = q.shape[1]
    group_size = num_query_heads // num_key_value_heads
    # TODO: Find more efficient way to do GQA.
    k = repeat(k, "B H T D -> B (H group_size) T D", group_size=group_size)
    v = repeat(v, "B H T D -> B (H group_size) T D", group_size=group_size)

    attention_scores = jnp.einsum(
        "B H t D, B H T D -> B H t T", q, k
    )  # (batch_size, num_query_heads, seq_len, seq_len)

    # Normalize
    attention_scores = attention_scores / jnp.sqrt(q.shape[-1])

    # Apply mask
    # TODO: Improve the efficiency.
    #       Applying mask after the attention scores are computed is not efficient.
    #       Further applying sliding window mask won't save resources either.
    attention_scores = jnp.where(mask[:, None, :, :], attention_scores, MASK_VALUE)

    # Softmax
    attention_weights = jax.nn.softmax(attention_scores, axis=-1).astype(k.dtype)

    # Apply attention weights
    output = jnp.einsum("B H t T, B H T D -> B H t D", attention_weights, v)

    return output


class MultiHeadAttention(nn.Module):
    num_query_heads: int
    num_key_value_heads: int
    hidden_size: int
    head_dim: int
    use_qk_norm: bool = False
    rope_theta: int = 10000
    rope_scale_factor: float = 1.0
    initializer: nn.initializers.Initializer = nn.initializers.uniform()

    def setup(self):
        self.q_proj = self.param(
            "q_proj", self.initializer,
            (self.num_query_heads, self.hidden_size, self.head_dim)
        )
        self.kv_proj = self.param(
            "kv_proj", self.initializer,
            (2, self.num_key_value_heads, self.hidden_size, self.head_dim)
        )
        self.o_proj = self.param(
            "o_proj", self.initializer,
            (self.num_query_heads, self.head_dim, self.hidden_size)
        )
        if self.use_qk_norm:
            self.k_norm = RMSNorm()
            self.q_norm = RMSNorm()

    def __call__(
        self, x: jax.Array, mask: jax.Array, position: jax.Array = None
    ) -> jax.Array:
        """
        Compute the multi-head attention.

        Args:
            x (jax.Array): Input tensor of shape (batch_size, seq_len, hidden_size).
            mask (jax.Array): Boolean mask tensor of shape (batch_size, seq_len, seq_len).
            position (jax.Array): Position indices of shape (batch_size, seq_len).

        Returns:
            jax.Array: The output tensor after applying attention, of shape (batch_size,
            seq_len, hidden_size).
        """
        # Project
        q = jnp.einsum("B T D, N D H -> B T N H", x, self.q_proj)
        k, v = jnp.einsum("B T D, C K D H -> C B T K H", x, self.kv_proj)

        # Apply qk norm
        if self.use_qk_norm:
            k = self.k_norm(k)
            q = self.q_norm(q)

        # Apply rope
        if position is not None:
            q = apply_rope(q, position, base=self.rope_theta, scale_factor=self.rope_scale_factor)
            k = apply_rope(k, position, base=self.rope_theta, scale_factor=self.rope_scale_factor)
        # TODO: make it configurable
        q = q * (self.head_dim**-0.5)

        # Transpose
        q = q.transpose(0, 2, 1, 3)  # (batch_size, num_query_heads, seq_len, head_dim)
        k = k.transpose(0, 2, 1, 3)  # (batch_size, num_key_value_heads, seq_len, head_dim)
        v = v.transpose(0, 2, 1, 3)  # (batch_size, num_key_value_heads, seq_len, head_dim)

        output = scaled_dot_product_attention(k, v, q, mask)  # (batch_size, num_query_heads, seq_len, head_dim)

        return jnp.einsum("B N T H, N H D -> B T D", output, self.o_proj)
