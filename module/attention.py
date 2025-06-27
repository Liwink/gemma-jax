"""Attention module"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from einops import repeat
from .rope_simple import apply_rope

MASK_VALUE = -1e10


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
    k = repeat(k, "b h t d -> b (h group_size) t d", group_size=group_size)
    v = repeat(v, "b h t d -> b (h group_size) t d", group_size=group_size)

    attention_scores = jnp.einsum(
        "b h t d, b h T d -> b h t T", q, k
    )  # (batch_size, num_query_heads, seq_len, seq_len)

    # Normalize
    attention_scores = attention_scores / jnp.sqrt(q.shape[-1])

    # Apply mask
    # TODO: Improve the efficiency.
    #       Applying mask after the attention scores are computed is not efficient.
    #       Further applying sliding window mask won't save resources either.
    attention_scores = jnp.where(mask[:, None, :, :], attention_scores, MASK_VALUE)

    # Softmax
    attention_weights = jax.nn.softmax(attention_scores, axis=-1)

    # Apply attention weights
    output = jnp.einsum("b h t T, b h T d -> b h t d", attention_weights, v)

    return output


class MultiHeadAttention(nn.Module):
    num_query_heads: int
    num_key_value_heads: int
    head_dim: int
    use_kv_norm: bool = False

    def setup(self):
        self.q_proj = nn.Dense(
            self.num_query_heads * self.head_dim, use_bias=False, name="q_proj"
        )
        self.k_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=False, name="k_proj"
        )
        self.v_proj = nn.Dense(
            self.num_key_value_heads * self.head_dim, use_bias=False, name="v_proj"
        )
        self.o_proj = nn.Dense(
            self.num_query_heads * self.head_dim, use_bias=False, name="o_proj"
        )
        if self.use_kv_norm:
            self.k_norm = RMSNorm()
            self.v_norm = RMSNorm()

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
        batch_size, seq_len, hidden_size = x.shape
        assert (
            hidden_size == self.num_query_heads * self.head_dim
        ), "Hidden size must be equal to num_query_heads * head_dim"

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape
        q = q.reshape(batch_size, seq_len, self.num_query_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        # Apply kv norm
        if self.use_kv_norm:
            k = self.k_norm(k)
            v = self.v_norm(v)

        # Apply rope
        if position is not None:
            q = apply_rope(q, position)
            k = apply_rope(k, position)

        # Transpose
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        output = scaled_dot_product_attention(k, v, q, mask)

        # Transpose and reshape back
        output = output.transpose(
            0, 2, 1, 3
        )  # (batch_size, num_query_heads, seq_len, head_dim)
        output = output.reshape(
            batch_size, seq_len, self.num_query_heads * self.head_dim
        )

        return self.o_proj(output)
