import jax
import flax.linen as nn


class MLP(nn.Module):
    hidden_size: int
    ffn_dim: int

    def setup(self):
        self.gate_proj = nn.Dense(self.ffn_dim, use_bias=False, name="gate_proj")
        self.up_proj = nn.Dense(self.ffn_dim, use_bias=False, name="up_proj")
        self.down_proj = nn.Dense(self.hidden_size, use_bias=False, name="down_proj")

    def __call__(
        self, x: jax.Array  # (batch_size, seq_len, hidden_size)
    ) -> jax.Array:  # (batch_size, seq_len, hidden_size)
        """
        gated = GELU(x @ W_gate) * (x @ W_up) # (batch_size, seq_len, ffn_dim)
        output = (gated @ W_down) # (batch_size, seq_len, hidden_size)
        """
        gate = nn.gelu(self.gate_proj(x))  # (batch_size, seq_len, ffn_dim)
        up = self.up_proj(x)  # (batch_size, seq_len, ffn_dim)
        gated = gate * up  # (batch_size, seq_len, ffn_dim)
        output = self.down_proj(gated)  # (batch_size, seq_len, hidden_size)

        return output
