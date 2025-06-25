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
* [ ] Full Transformer model
* [ ] Sliding window attention
* [x] Grouped-Query Attention
* [ ] Token embedding
* [ ] Output layer
* [ ] Config loading
* [ ] Weight loading
* [ ] Inference, generating next token
* [ ] Inference benchmark
* [ ] High performance inference

### Potential Efficiency Improvement

List some of the current inefficient implementation.

* Applying mask after the attention scores are computed is not efficient.
* The same applies to the sliding window mask.
* Explore flash attention.
* More efficient GQA implementation, without repeating the key and value
  heads for each query head.
