import jax.numpy as jnp
from module.transformer import Transformer
from models.gemma3 import load_gemma3_params, GEMMA_3_1B_CONFIG
import sentencepiece as spm
import os

GEMMA_3_1B_IT_PATH = os.path.expanduser(
    "~/Models/gemma-3-flax-gemma3-1b-it-v1/gemma3-1b-it"
)
GEMMA_3_1B_PATH = os.path.expanduser("~/Models/gemma-3-flax-gemma3-1b-v1/gemma3-1b")
TOKENIZER_PATH = os.path.expanduser(
    "~/Models/gemma-3-flax-gemma3-1b-it-v1/tokenizer.model"
)


class TestGemma3:
    def test_gemma3_basics(self):
        model = Transformer(config=GEMMA_3_1B_CONFIG)
        params = load_gemma3_params(path=GEMMA_3_1B_PATH)
        tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)

        text = "I'm a language model, "
        tokens = jnp.array([tokenizer.encode(text, add_bos=True)], dtype=jnp.int32)
        logits = model.apply({"params": params}, tokens)
        assert logits.shape == (1, tokens.shape[1], GEMMA_3_1B_CONFIG.vocab_size)


if __name__ == "__main__":
    model = Transformer(config=GEMMA_3_1B_CONFIG)
    params = load_gemma3_params(path=GEMMA_3_1B_PATH)

    tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    text = "Eiffel tower is located in"
    for i in range(100):
        tokens = jnp.array([tokenizer.encode(text, add_bos=True)], dtype=jnp.int32)
        logits = model.apply({"params": params}, tokens)
        _id = logits[0, -1, :].argmax(axis=-1).tolist()
        text += tokenizer.decode_ids(_id)
        print("\r" + text, end="", flush=True)
