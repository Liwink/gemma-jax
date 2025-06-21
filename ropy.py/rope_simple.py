import jax
import jax.numpy as jnp
from einops import rearrange

_DEFAULT_BASE = 10000


def apply_rope(
        input: jax.Array, # (batch_size, seq_len, head_dim)
        position: jax.Array, # (batch_size, seq_len)
        base: int = _DEFAULT_BASE,
    ) -> jax.Array:
    """
    Apply rotary positional embeddings (RoPE) to the input tensor.
    
    Args:
        input: Input tensor of shape (batch_size, seq_len, head_dim)
        position: Position indices of shape (batch_size, seq_len)
        base: Base frequency for the positional embeddings (default: 10000.0)
    
    Returns:
        Tensor of same shape as input with rotary positional embeddings applied
    """
    head_dim = input.shape[-1]
    # get freq
    power = -2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = base ** power
    freq = position[..., None] * timescale

    cos_freq = jnp.cos(freq)
    sin_freq = jnp.sin(freq)

    # split into even and odd indices
    x_even = input[..., ::2] # (batch_size, seq_len, head_dim // 2)
    x_odd = input[..., 1::2] # (batch_size, seq_len, head_dim // 2)

    # apply rotation
    x_even, x_odd = (
        x_even * cos_freq - x_odd * sin_freq,
        x_odd * cos_freq + x_even * sin_freq
    )

    # concatenate
    stacked = jnp.stack([x_even, x_odd], axis=-1) # (batch_size, seq_len, head_dim // 2, 2)
    return rearrange(stacked, "... d 2 -> ... (2 d)")
