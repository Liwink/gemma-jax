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
* [ ] Transformer block
* [ ] Full Transformer model
* [ ] Sliding window attention
* [ ] Grouped-Query Attention
* [ ] Token embedding
* [ ] Output layer
* [ ] Config loading
* [ ] Weight loading
* [ ] Inference, generating next token
* [ ] Inference benchmark
* [ ] High performance inference
