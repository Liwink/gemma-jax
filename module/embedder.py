import jax
import jax.numpy as jnp
import flax.linen as nn

class Embedder(nn.Module):
    vocab_size: int
    hidden_size: int

    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.hidden_size)

    def encode(self, x: jax.Array) -> jax.Array:
        x = self.token_embedding(x)
        x *= jnp.sqrt(self.hidden_size).astype(x.dtype)
        return x

    def decode(self, x: jax.Array) -> jax.Array:
        return self.token_embedding.attend(x)