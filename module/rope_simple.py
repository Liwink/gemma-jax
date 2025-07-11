"""Apply rotary positional embeddings (RoPE) to the input tensor."""

import jax
import jax.numpy as jnp
_DEFAULT_BASE = 10000
_DEFAULT_SCALE_FACTOR = 1.0


def apply_rope(
    x: jax.Array,  # (batch_size, seq_len, num_heads, head_dim)
    position: jax.Array,  # (batch_size, seq_len)
    base: int = _DEFAULT_BASE,
    scale_factor: float = _DEFAULT_SCALE_FACTOR,
    middle_split: bool = True,
    layer: int = 0,
) -> jax.Array:
    """
    Apply rotary positional embeddings (RoPE) to the input tensor.

    Args:
        x: Input tensor of shape (batch_size, seq_len, num_heads, head_dim)
        position: Position indices of shape (batch_size, seq_len)
        base: Base frequency for the positional embeddings (default: 10000.0)
        scale_factor: Scale factor for the positional embeddings (default: 1.0)
    Returns:
        Tensor of same shape as input with rotary positional embeddings applied
    """
    head_dim = x.shape[-1]
    assert head_dim % 2 == 0
    # get freq
    power = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = base**power
    freq = (
        position[..., None] / timescale[None, None, :]
    )  # (batch_size, seq_len, head_dim // 2)

    if scale_factor > 1.0:
        freq = freq / scale_factor

    # expand freq to (batch_size, seq_len, num_heads, head_dim // 2)
    freq = freq[..., None, :]
    cos_freq = jnp.cos(freq)
    sin_freq = jnp.sin(freq)

    if middle_split:
        x_first, x_second = jnp.split(x, 2, axis=-1)
        first_part = x_first * cos_freq - x_second * sin_freq
        second_part = x_second * cos_freq + x_first * sin_freq

        return jnp.concatenate([first_part, second_part], axis=-1).astype(x.dtype)

    # split into even and odd indices
    x_even = x[..., ::2]  # (batch_size, seq_len, num_heads, head_dim // 2)
    x_odd = x[..., 1::2]  # (batch_size, seq_len, num_heads, head_dim // 2)

    # apply rotation
    x_even, x_odd = (
        x_even * cos_freq - x_odd * sin_freq,
        x_odd * cos_freq + x_even * sin_freq,
    )

    stacked = jnp.stack(
        [x_even, x_odd], axis=-1
    )  # (batch_size, seq_len, num_heads, head_dim // 2, 2)
    return stacked.reshape(*stacked.shape[:-2], -1).astype(x.dtype)
