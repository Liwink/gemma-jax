import jax
import jax.numpy as jnp
import flax.linen as nn

class RMSNorm(nn.Module):
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        # TODO: use the scale parameter
        rms = jnp.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        return x * rms
