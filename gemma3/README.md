# Gemma3 Checkpoint Loading

Load Gemma 3 checkpoints.

## Architecture

The architecture config of the model is as follows:

```json
{
  "architectures": [
    "Gemma3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attn_logit_softcapping": null,
  "bos_token_id": 2,
  "cache_implementation": "hybrid",
  "eos_token_id": [
    1,
    106
  ],
  "final_logit_softcapping": null,
  "head_dim": 256,
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 1152,
  "initializer_range": 0.02,
  "intermediate_size": 6912,
  "max_position_embeddings": 32768,
  "model_type": "gemma3_text",
  "num_attention_heads": 4,
  "num_hidden_layers": 26,
  "num_key_value_heads": 1,
  "pad_token_id": 0,
  "query_pre_attn_scalar": 256,
  "rms_norm_eps": 1e-06,
  "rope_local_base_freq": 10000,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": 512,
  "sliding_window_pattern": 6,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.52.4",
  "use_cache": true,
  "vocab_size": 262144
}
```

## Checkpoint Format

`gemma3-1b-it` downloaded from kaggle for the official FLAX implementation has the following format:

Parameter keys:

```text
transformer/embedder
    {'input_embedding': ArrayMetadata: 
        name=transformer/embedder.input_embedding, 
        shape=(262144, 1152), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(262144, 1152), write_shape=None)
    }

transformer/final_norm
    {'scale': ArrayMetadata: 
        name=transformer/final_norm.scale, 
        shape=(1152,), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(1152,), write_shape=None)
    }

transformer/layer_0/attn/_key_norm
    {'scale': ArrayMetadata: 
        name=transformer/layer_0/attn/_key_norm.scale, 
        shape=(256,), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(256,), write_shape=None)
    }

transformer/layer_0/attn/_query_norm
    {'scale': ArrayMetadata: 
        name=transformer/layer_0/attn/_query_norm.scale, 
        shape=(256,), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(256,), write_shape=None)
    }

transformer/layer_0/attn/attn_vec_einsum
    {'w': ArrayMetadata: 
        name=transformer/layer_0/attn/attn_vec_einsum.w, 
        shape=(4, 256, 1152), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(4, 256, 1152), write_shape=None)
    }

transformer/layer_0/attn/kv_einsum
    {'w': ArrayMetadata: 
        name=transformer/layer_0/attn/kv_einsum.w, 
        shape=(2, 1, 1152, 256), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(2, 1, 1152, 256), write_shape=None)
    }

transformer/layer_0/attn/q_einsum
    {'w': ArrayMetadata: 
        name=transformer/layer_0/attn/q_einsum.w, 
        shape=(4, 1152, 256), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(4, 1152, 256), write_shape=None)
    }

transformer/layer_0/mlp/gating_einsum
    {'w': ArrayMetadata: 
        name=transformer/layer_0/mlp/gating_einsum.w, 
        shape=(2, 6912, 1152), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(2, 6912, 1152), write_shape=None)
    }

transformer/layer_0/mlp/linear
    {'w': ArrayMetadata: 
        name=transformer/layer_0/mlp/linear.w, 
        shape=(6912, 1152), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(6912, 1152), write_shape=None)
    }

transformer/layer_0/post_attention_norm
    {'scale': ArrayMetadata: 
        name=transformer/layer_0/post_attention_norm.scale, 
        shape=(1152,), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(1152,), write_shape=None)
    }

transformer/layer_0/post_ffw_norm
    {'scale': ArrayMetadata: 
        name=transformer/layer_0/post_ffw_norm.scale, 
        shape=(1152,), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(1152,), write_shape=None)
    }

transformer/layer_0/pre_attention_norm
    {'scale': ArrayMetadata: 
        name=transformer/layer_0/pre_attention_norm.scale, 
        shape=(1152,), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(1152,), write_shape=None)
    }

transformer/layer_0/pre_ffw_norm
    {'scale': ArrayMetadata: 
        name=transformer/layer_0/pre_ffw_norm.scale, 
        shape=(1152,), 
        sharding=None, 
        dtype=bfloat16, 
        storage=StorageMetadata(chunk_shape=(1152,), write_shape=None)
    }
```
