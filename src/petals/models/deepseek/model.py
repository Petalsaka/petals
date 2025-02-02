"""DeepSeek model classes for distributed inference"""

from typing import Optional

import torch
import torch.nn as nn
from hivemind import DHT
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama import LlamaPreTrainedModel

from petals.client.from_pretrained import FromPretrainedMixin
from petals.client.lm_head import LMHead
from petals.client.ptune import PTuneMixin
from petals.client.remote_generation import RemoteGenerationMixin, RemotePastKeyValues
from petals.client.remote_sequential import RemoteSequential
from petals.models.deepseek.config import DistributedDeepSeekConfig


class DistributedDeepSeekModel(FromPretrainedMixin, PTuneMixin, LlamaPreTrainedModel):
    """DeepSeek model with distributed transformer layers"""

    _keys_to_ignore_on_load_missing = PTuneMixin._keys_to_ignore_on_load_missing
    _keys_to_ignore_on_load_unexpected = [r"^model\.layers\."]

    config_class = DistributedDeepSeekConfig

    def __init__(self, config: DistributedDeepSeekConfig, *, dht: Optional[DHT] = None):
        n_layer, config.num_hidden_layers = config.num_hidden_layers, 0  # Prevent initialization
        super().__init__(config)
        config.num_hidden_layers = n_layer

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = RemoteSequential(config, dht=dht)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.requires_grad_(False)  # Forbid accumulate grads for embeddings and layernorm
        self.init_prompts(config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[RemotePastKeyValues] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        output_shape = input_shape + (hidden_states.size(-1),)

        hidden_states = self.layers(hidden_states)

        if past_key_values is None:
            past_key_values = RemotePastKeyValues()
        past_key_values.update_seen(hidden_states.size(1))

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(output_shape)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
        )


class DistributedDeepSeekForCausalLM(FromPretrainedMixin, RemoteGenerationMixin, LlamaPreTrainedModel):
    """DeepSeek model for causal language modeling with distributed transformer layers"""

    _keys_to_ignore_on_load_missing = DistributedDeepSeekModel._keys_to_ignore_on_load_missing + [r"^lm_head\."]
    _keys_to_ignore_on_load_unexpected = DistributedDeepSeekModel._keys_to_ignore_on_load_unexpected

    config_class = DistributedDeepSeekConfig

    def __init__(self, config: DistributedDeepSeekConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = DistributedDeepSeekModel(config)
        self.lm_head = LMHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    @property
    def transformer(self) -> DistributedDeepSeekModel:  # For compatibility with RemoteGenerationMixin
        return self.model 