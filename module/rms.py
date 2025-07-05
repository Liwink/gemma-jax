import jax
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Apply RMSNorm to the last dimension of the input.
        """

        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))

        r_rms = jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

        return x * r_rms * (scale + 1)
