import jax
import jax.numpy as jnp
from module.transformer import Transformer
from module.config import GEMMA_3_1B_CONFIG
from models.gemma3 import load_gemma3_params

GEMMA_3_1B_PATH = "/Users/liuyihe/Models/gemma-3-flax-gemma3-1b-it-v1/gemma3-1b-it"


class TestGemma3:
    def test_gemma3_basics(self):
        model = Transformer(config=GEMMA_3_1B_CONFIG)
        params = load_gemma3_params(path=GEMMA_3_1B_PATH)

        dummy_tokens = jnp.ones((1, 10), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, 10, 10), dtype=jnp.bool_)

        output = model.apply({"params": params}, dummy_tokens, dummy_mask)

        assert output.shape == (1, 10, GEMMA_3_1B_CONFIG.vocab_size)
