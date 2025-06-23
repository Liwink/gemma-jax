"""Attention module"""

import jax
import jax.numpy as jnp
import flax.linen as nn

MASK_VALUE = -1e10


def scaled_dot_product_attention(
    k: jax.Array,  # (batch_size, num_heads, seq_len, head_dim)
    v: jax.Array,  # (batch_size, num_heads, seq_len, head_dim)
    q: jax.Array,  # (batch_size, num_heads, seq_len, head_dim)
    mask: jax.Array,  # (batch_size, seq_len, seq_len)
) -> jax.Array:  # (batch_size, num_heads, seq_len, head_dim)
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

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int

    def setup(self):
        self.q_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name="q_proj")
        self.k_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name="v_proj")
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, use_bias=False, name="o_proj")
    
    def __call__(self, x: jax.Array, mask: jax.Array) -> jax.Array:
        """
        Compute the multi-head attention.

        Args:
            x (jax.Array): Input tensor of shape (batch_size, seq_len, hidden_size).
            mask (jax.Array): Boolean mask tensor of shape (batch_size, seq_len, seq_len).

        Returns:
            jax.Array: The output tensor after applying attention, of shape (batch_size,
            seq_len, hidden_size).
        """
        batch_size, seq_len, hidden_size = x.shape

        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape and transpose
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        output = scaled_dot_product_attention(k, v, q, mask)

        # Transpose and reshape back
        output = output.transpose(0, 2, 1, 3) # (batch_size, seq_len, num_heads, head_dim)
        output = output.reshape(batch_size, seq_len, self.num_heads * self.head_dim)
        
        return self.o_proj(output)
