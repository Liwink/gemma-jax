import jax
import jax.numpy as jnp
from module.transformer import Transformer
from models.gemma3 import load_gemma3_params, GEMMA_3_1B_CONFIG
import sentencepiece as spm

GEMMA_3_1B_IT_PATH = "/Users/liuyihe/Models/gemma-3-flax-gemma3-1b-it-v1/gemma3-1b-it"
GEMMA_3_1B_PATH = "/Users/liuyihe/Models/gemma-3-flax-gemma3-1b-v1/gemma3-1b"
TOKENIZER_PATH = "/Users/liuyihe/Models/gemma-3-flax-gemma3-1b-it-v1/tokenizer.model"

def generate(model, params, tokenizer, text):
    tokens = jnp.array([tokenizer.encode_as_ids(text)], dtype=jnp.int32)
    mask = jnp.tril(jnp.ones((1, tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_))
    position = jnp.array([list(range(tokens.shape[1]))], dtype=jnp.int32)
    
    logits = model.apply({"params": params}, tokens, mask, position)
    return tokenizer.decode_ids(logits[0].argmax(axis=-1).tolist())
    

class TestGemma3:
    def test_gemma3_basics(self):
        model = Transformer(config=GEMMA_3_1B_CONFIG)
        params = load_gemma3_params(path=GEMMA_3_1B_PATH)

        tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
        text = "I'm a language model, "
        tokens = jnp.array([tokenizer.encode_as_ids(text)], dtype=jnp.int32)
        mask = jnp.tril(jnp.ones((1, tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_))
        position = jnp.array([list(range(tokens.shape[1]))], dtype=jnp.int32)

        logits = model.apply({"params": params}, tokens, mask, position)
        assert logits.shape == (1, tokens.shape[1], GEMMA_3_1B_CONFIG.vocab_size)


if __name__ == "__main__":
    model = Transformer(config=GEMMA_3_1B_CONFIG)
    params = load_gemma3_params(path=GEMMA_3_1B_PATH)

    tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
    # text = "Hello"
    text = "Eiffel tower is located in"
    for i in range (5):
        tokens = jnp.array([tokenizer.encode(text, add_bos=True)], dtype=jnp.int32)
        mask = jnp.tril(jnp.ones((1, tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_))
        position = jnp.array([list(range(tokens.shape[1]))], dtype=jnp.int32)

        logits = model.apply({"params": params}, tokens, mask, position)
        _id = logits[0, -1, :].argmax(axis=-1).tolist()
        text += tokenizer.decode_ids(_id)
        print(text)
