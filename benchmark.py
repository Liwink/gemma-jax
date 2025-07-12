
import jax.numpy as jnp
from module.transformer import Transformer
from models.gemma3 import load_gemma3_params, GEMMA_3_1B_CONFIG
import os
import time
import argparse
import random

GEMMA_3_1B_PATH = os.path.expanduser("~/Models/gemma-3-flax-gemma3-1b-v1/gemma3-1b")

BATCH_SIZE = 64
NUM_INPUT_TOKENS = 100
NUM_OUTPUT_TOKENS = 100


def apply(model, params, tokens):
    mask = jnp.tril(
        jnp.ones((tokens.shape[0], tokens.shape[1], tokens.shape[1]), dtype=jnp.bool_)
    )
    position = jnp.array([list(range(tokens.shape[1]))] * tokens.shape[0], dtype=jnp.int32)
    return model.apply({"params": params}, tokens, mask, position)


def main(batch_size: int = BATCH_SIZE, num_input_tokens: int = NUM_INPUT_TOKENS, num_output_tokens: int = NUM_OUTPUT_TOKENS):
    """
    Benchmark the inference speed of the model.
    """
    model = Transformer(config=GEMMA_3_1B_CONFIG)
    params = load_gemma3_params(path=GEMMA_3_1B_PATH)

    token_ids = [[random.randint(0, GEMMA_3_1B_CONFIG.vocab_size) for _ in range(num_input_tokens)] for _ in range(batch_size)]

    start_time = time.time()

    for _ in range(num_output_tokens):
        tokens_batch = jnp.array(token_ids, dtype=jnp.int32)
        
        logits = model.apply({"params": params}, tokens_batch)
        
        last_token_logits = []
        for i, t in enumerate(token_ids):
            last_token_logits.append(logits[i, len(t)-1, :])
        
        next_ids = jnp.array(last_token_logits).argmax(axis=-1).tolist()

        for i in range(len(token_ids)):
            token_ids[i].append(next_ids[i])

    end_time = time.time()

    total_time = end_time - start_time
    total_tokens_generated = (num_input_tokens + num_output_tokens) * batch_size
    tokens_per_second = total_tokens_generated / total_time

    print(f"--- Benchmark Results ---")
    print(f"Batch size: {batch_size}")
    print(f"Number of input tokens: {num_input_tokens}")
    print(f"Number of output tokens: {num_output_tokens}")
    print(f"Total tokens generated: {total_tokens_generated}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Tokens per second: {tokens_per_second:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma JAX Inference Benchmark")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size for inference.")
    parser.add_argument("--num_input_tokens", type=int, default=100, help="The number of input tokens for each sequence in the batch.")
    parser.add_argument("--num_output_tokens", type=int, default=100, help="The number of output tokens for each sequence in the batch.")
    args = parser.parse_args()
    
    main(args.batch_size, args.num_input_tokens, args.num_output_tokens)
