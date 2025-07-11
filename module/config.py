from dataclasses import dataclass

@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int
    hidden_size: int
    ffn_dim: int
    num_query_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: int
    use_qk_norm: bool

