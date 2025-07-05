import jax
from .attention import MultiHeadAttention
from .mlp import MLP
from .rms import RMSNorm
import flax.linen as nn


class Block(nn.Module):
    hidden_size: int
    ffn_dim: int
    num_query_heads: int
    num_key_value_heads: int
    head_dim: int
    use_qk_norm: bool = False
    rope_theta: int = 10000
    rope_scale_factor: float = 1.0
    layer: int = 0

    def setup(self):
        self.attention = MultiHeadAttention(
            self.num_query_heads,
            self.num_key_value_heads,
            self.hidden_size,
            self.head_dim,
            self.use_qk_norm,
            self.rope_theta,
            self.rope_scale_factor,
        )
        self.mlp = MLP(self.hidden_size, self.ffn_dim, layer=self.layer)
        self.pre_attention_norm = RMSNorm()
        self.post_attention_norm = RMSNorm()
        self.pre_ffw_norm = RMSNorm()
        self.post_ffw_norm = RMSNorm()

    def __call__(
        self, x: jax.Array, mask: jax.Array, position: jax.Array = None
    ) -> jax.Array:
        """
        Apply the transformer block.
        Gemma 3 uses both pre-norm and post-norm with RSMNorm.

        Args:
            x (jax.Array): Input tensor of shape (batch_size, seq_len, hidden_size).
            mask (jax.Array): Boolean mask tensor of shape (batch_size, seq_len, seq_len).
            position (jax.Array): Position indices of shape (batch_size, seq_len).

        Returns:
            jax.Array: The output tensor after applying the transformer block, of shape (batch_size,
            seq_len, hidden_size).
        """
        # Apply pre-norm and multi-head attention
        attn_output = self.attention(self.pre_attention_norm(x), mask, position)
        # Residual connection
        attn_output = self.post_attention_norm(attn_output)
        x = x + attn_output
        # Apply pre-norm and MLP

        normed_x = self.pre_ffw_norm(x)
        mlp_output = self.mlp(normed_x)
        mlp_output = self.post_ffw_norm(mlp_output)
        # Residual connection
        x = x + mlp_output
        return x
