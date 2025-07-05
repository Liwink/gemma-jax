from module.config import TransformerConfig
import orbax.checkpoint as ocp

GEMMA_3_1B_CONFIG = TransformerConfig(
    vocab_size=262144,
    hidden_size=1152,
    ffn_dim=6912,
    num_query_heads=4,
    num_key_value_heads=1,
    num_hidden_layers=26,
    head_dim=256,
    rms_norm_eps=1e-6,
    rope_theta=10000,
    use_qk_norm=True,
)


def load_gemma3_params(
    path: str,
    config: TransformerConfig = GEMMA_3_1B_CONFIG,
) -> dict:
    checkpointer = ocp.PyTreeCheckpointer()
    raw_params = checkpointer.restore(path)

    model_params = {
        "embedder": {
            "token_embedding": {
                "embedding": raw_params["transformer/embedder"]["input_embedding"]
            }
        },
        "final_norm": {"scale": raw_params["transformer/final_norm"]["scale"]},
    }

    for i in range(config.num_hidden_layers):
        layer_name = f"transformer/layer_{i}"
        block_params = {
            "pre_attention_norm": {
                "scale": raw_params[f"{layer_name}/pre_attention_norm"]["scale"]
            },
            "post_attention_norm": {
                "scale": raw_params[f"{layer_name}/post_attention_norm"]["scale"]
            },
            "pre_ffw_norm": {
                "scale": raw_params[f"{layer_name}/pre_ffw_norm"]["scale"]
            },
            "post_ffw_norm": {
                "scale": raw_params[f"{layer_name}/post_ffw_norm"]["scale"]
            },
            "attention": {
                "q_proj": raw_params[f"{layer_name}/attn/q_einsum"]["w"],
                "kv_proj": raw_params[f"{layer_name}/attn/kv_einsum"]["w"],
                "o_proj": raw_params[f"{layer_name}/attn/attn_vec_einsum"]["w"],
                "q_norm": {
                    "scale": raw_params[f"{layer_name}/attn/_query_norm"]["scale"]
                },
                "k_norm": {
                    "scale": raw_params[f"{layer_name}/attn/_key_norm"]["scale"]
                },
            },
            "mlp": {
                "gate_proj": raw_params[f"{layer_name}/mlp/gating_einsum"]["w"][0].T,
                "up_proj": raw_params[f"{layer_name}/mlp/gating_einsum"]["w"][1].T,
                "gating_proj": raw_params[f"{layer_name}/mlp/gating_einsum"]["w"],
                "down_proj": raw_params[f"{layer_name}/mlp/linear"]["w"],
            },
        }
        model_params[f"blocks_{i}"] = block_params

    return model_params
