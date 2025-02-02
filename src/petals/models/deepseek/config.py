"""Configuration classes for DeepSeek models."""

import os
import logging
from typing import Optional, Union

from hivemind import get_logger
from transformers.configuration_utils import PretrainedConfig

from petals.client.config import ClientConfig
from petals.client.lm_head import LMHeadConfig
from petals.client.ptune import PTuneConfig

# Set default logging level to WARNING
logging.getLogger("petals.models.deepseek").setLevel(logging.WARNING)
logger = get_logger(__name__)

class DistributedDeepSeekConfig(PretrainedConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    # This configuration supports DeepSeek-R1 which uses the deepseek_v3 model type
    model_type = "deepseek_v3"
    block_prefix = "model.layers"

    def __init__(
        self,
        vocab_size=129280,
        hidden_size=7168,
        intermediate_size=18432,
        num_hidden_layers=61,
        num_attention_heads=128,
        num_key_value_heads=128,
        hidden_act="silu",
        max_position_embeddings=163840,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=False,
        attention_bias=False,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        rope_theta=10000,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def attn_class(self):
        from petals.models.deepseek.block import DeepSeekAttention
        return DeepSeekAttention

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"

        
        result = super().from_pretrained(model_name_or_path, *args, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        config.dht_prefix = dht_prefix
        return result

# Import after class definition to avoid circular dependency
from petals.models.deepseek.wrapped_block import WrappedDeepSeekBlock
DistributedDeepSeekConfig.block_class = WrappedDeepSeekBlock 

try:
    from transformers import DeepseekConfig, DeepseekModel
except ImportError:
    DeepseekConfig = None
    DeepseekModel = None 