# Transformer in JAX

This is a simple implementation of transformers in JAX.
The goal is to load Gemma 3 model and run it locally.

## Gemma 3 Architecture

Highlight the Gemma 3 key architecture differences from the original transformers.

* Grouped-Query Attention
* Post-norm and pre-norm with RMSNorm
* 5:1 interleaving of local sliding window attention and global attention
* RoPE
* GeGLU activation

## TODO

* [x] RoPE
* [x] Multihead attention
* [x] RMSNorm
* [x] GeGLU activation
* [x] MLP
* [x] Transformer block
* [x] Full Transformer model
* [ ] Sliding window attention
* [x] Grouped-Query Attention
* [x] Token embedding
* [x] Output layer
* [x] Config loading
* [x] Weight loading
* [x] Inference, generating next token
* [ ] Inference benchmark
* [ ] High performance inference

### Potential Efficiency Improvement

List some of the current inefficient implementation.

* Applying mask after the attention scores are computed is not efficient.
* The same applies to the sliding window mask.
* Explore flash attention.
* More efficient GQA implementation, without repeating the key and value
  heads for each query head.

## Get Started

```python
model = Transformer(config=GEMMA_3_1B_CONFIG)
params = load_gemma3_params(path=GEMMA_3_1B_PATH)
tokenizer = spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)
tokens = jnp.array([tokenizer.encode(text, add_bos=True)])
logits = model.apply({"params": params}, tokens)
next_token_id = logits[0, -1, :].argmax(axis=-1).tolist()
text += tokenizer.decode_ids(next_token_id)
print(text)
```

## Debugging Notes

### Logging

* Use `jax.debug.print` to print the tensor values.
* Use `jax.debug.callback` to save the tensor values to disk.

```python
def save_metrics_callback(metrics, step=0, prefix="debug"):
    """Save metrics to disk via callback"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_step_{step}_{timestamp}.jnp"

    jnp.save(filename, metrics)
    print(f"Saved metrics to {filename}")

jax.debug.callback(save_metrics_callback, x, step=0, prefix="x")
```

### Numerical Discrepancy Between Equivalent jnp.einsum Formulations

[Reported issue](https://github.com/jax-ml/jax/issues/29990)
[Repro](https://colab.research.google.com/drive/1latj_SynZyqWxCKnwlhTegD1RMkRwpNP#scrollTo=KK0eDUboSzDY)

The final logits of this model are different from the official FLAX implementation.
After tracing the model, the discrepancy starts in the 2nd block MLP.
Our implementation used separate einsum for gating and up projection, while the offical implementation
uses a combined einsum.

### Fix Norm Epsilon Discrepancy

The official implementation uses `eps=1e-6`, while our implementation uses `eps=1e-5`.

The following table shows the number of different elements in the input array between the official and our implementation at the first 10 blocks, i.e., `(input_x_official != input_x_ours).sum()`

| Index | Before | After |
|-------|--------|-------|
| 0     | 0      | 0     |
| 1     | 0      | 0     |
| 2     | 0      | 0     |
| 3     | 0      | 0     |
| 4     | 644    | 0     |
| 5     | 1043   | 0     |
| 6     | 1100   | 0     |
| 7     | 1131   | 0     |
| 8     | 1125   | 0     |
| 9     | 1132   | 4     |
| 10    | 1131   | 27    |
| 11    | 1129   | 153   |
| 12    | 1129   | 444   |
| 13    | 1143   | 513   |
| 14    | 1146   | 579   |
| 15    | 1140   | 644   |
| 16    | 1147   | 541   |
| 17    | 1137   | 682   |
| 18    | 1145   | 780   |
| 19    | 1143   | 878   |

### Numerical Discrepancy is a Red Herring

The previous tests focused on the single token input to exclude Attention module varify the remaining modules.
I was trying to make the output logits identical for both our implementation and the official implementation.
However, the hidden state and logits numerical discrepancy turns out to be a red herring.
Even if they are not identical, the final results after softmax and top_k are the same.

However, if letting the our model to genearte multiple tokens, it starts to repeat the same tokens.
So I shifted the focus to the attention module, and quickly found an obvious bug.
The query tensor was being scaled twice, once in the scaled_dot_product_attention function, and once in the MultiHeadAttention module.
Once it's fixed, the model starts to generate meaningful tokens.
