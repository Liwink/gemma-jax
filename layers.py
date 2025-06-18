import jax
import jax.numpy as jnp
import flax.linen as nn

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

