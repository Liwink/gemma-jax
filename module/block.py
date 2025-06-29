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

    def setup(self):
        self.attention = MultiHeadAttention(
            self.num_query_heads,
            self.num_key_value_heads,
            self.hidden_size,
            self.head_dim,
            self.use_qk_norm,
        )
        self.mlp = MLP(self.hidden_size, self.ffn_dim)
        self.attn_pre_norm = RMSNorm()
        self.attn_post_norm = RMSNorm()
        self.mlp_pre_norm = RMSNorm()
        self.mlp_post_norm = RMSNorm()

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
        attn_output = self.attention(self.attn_pre_norm(x), mask, position)
        # Residual connection
        x = x + self.attn_post_norm(attn_output)
        # Apply pre-norm and MLP
        mlp_output = self.mlp(self.mlp_pre_norm(x))
        # Residual connection
        x = x + self.mlp_post_norm(mlp_output)
        return x
