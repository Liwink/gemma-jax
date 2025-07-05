import jax
import jax.numpy as jnp
import flax.linen as nn


class MLP(nn.Module):
    hidden_size: int
    ffn_dim: int
    initializer: nn.initializers.Initializer = nn.initializers.uniform()
    layer: int = 0

    def setup(self):
        self.gate_proj = self.param(
            "gate_proj", self.initializer,
            (self.hidden_size, self.ffn_dim)
        )
        self.up_proj = self.param(
            "up_proj", self.initializer,
            (self.hidden_size, self.ffn_dim)
        )
        self.down_proj = self.param(
            "down_proj", self.initializer,
            (self.ffn_dim, self.hidden_size)
        )

    def __call__(
        self, x: jax.Array  # (batch_size, seq_len, hidden_size)
    ) -> jax.Array:  # (batch_size, seq_len, hidden_size)
        """
        gated = GELU(x @ W_gate) * (x @ W_up) # (batch_size, seq_len, ffn_dim)
        output = (gated @ W_down) # (batch_size, seq_len, hidden_size)
        """
        g = jnp.einsum("B T D, D F -> B T F", x, self.gate_proj)
        gate = nn.gelu(g)  # (batch_size, seq_len, ffn_dim)
        up = jnp.einsum("B T D, D F -> B T F", x, self.up_proj)  # (batch_size, seq_len, ffn_dim)
        gated = gate * up  # (batch_size, seq_len, ffn_dim)
        output = jnp.einsum("B T F, F D -> B T D", gated, self.down_proj)  # (batch_size, seq_len, hidden_size)
        return output
