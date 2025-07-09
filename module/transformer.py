import jax
import jax.numpy as jnp
import flax.linen as nn
from .config import TransformerConfig
from .block import Block
from .embedder import Embedder
from .rms import RMSNorm


class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.embedder = Embedder(self.config.vocab_size, self.config.hidden_size)
        self.blocks = [
            Block(
                self.config.hidden_size,
                self.config.ffn_dim,
                self.config.num_query_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                self.config.use_qk_norm,
                # TODO: make it configurable
                rope_theta=10_000 if (i + 1) % 6 != 0 else 1_000_000,
                rope_scale_factor=1.0 if (i + 1) % 6 != 0 else 8.0,
                layer=i,
            )
            for i in range(self.config.num_hidden_layers)
        ]
        self.final_norm = RMSNorm()

    @nn.jit
    def __call__(
        self,
        tokens: jax.Array,
        mask: jax.Array | None = None,
        position: jax.Array | None = None,
    ) -> jax.Array:
        x = self.embedder.encode(tokens)
        B, T = tokens.shape
        if mask is None:
            mask = jnp.tril(jnp.ones((B, T, T), dtype=jnp.bool_))
        if position is None:
            position = jnp.array([list(range(T))])
        for block in self.blocks:
            x = block(x, mask, position)
        x = self.final_norm(x)
        return self.embedder.decode(x)
