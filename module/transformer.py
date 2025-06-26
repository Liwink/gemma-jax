import jax
from flax import nn
from .config import TransformerConfig
from .block import Block


class Transformer(nn.Module):
    config: TransformerConfig

    def setup(self):
        self.blocks = nn.ModuleList(
            [
                Block(
                    self.config.hidden_size,
                    self.config.ffn_dim,
                    self.config.num_query_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                )
                for _ in range(self.config.num_hidden_layers)
            ]
        )
