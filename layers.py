import jax
import jax.numpy as jnp
import flax.linen as nn

class RSMNorm(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        # TODO: use the scale parameter
        rms = jnp.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x * rms



class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        self.gate_proj = nn.Dense(intermediate_size, use_bias=False, name="gate_proj")
        self.up_proj = nn.Dense(intermediate_size, use_bias=False, name="up_proj")
        self.down_proj = nn.Dense(hidden_size, use_bias=False, name="down_proj")

    @nn.compact
    def __call__(self, x):
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        gated = gate * up
        output = self.down_proj(gated)

        return output

