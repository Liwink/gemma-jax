import jax
import flax.linen as nn
from .config import TransformerConfig
from .block import Block
from .embedder import Embedder
from .rms import RMSNorm

class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.embedder = Embedder(self.config.vocab_size, self.config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                Block(
                    self.config.hidden_size,
                    self.config.ffn_dim,
                    self.config.num_query_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    self.config.use_kv_norm,
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm()

    def __call__(
        self, tokens: jax.Array, mask: jax.Array, position: jax.Array = None
    ) -> jax.Array:
        x = self.embedder.encode(tokens)
        for block in self.blocks:
            x = block(x, mask, position)
        x = self.final_norm(x)
        return self.embedder.decode(x)
