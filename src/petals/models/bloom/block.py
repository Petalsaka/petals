"""
Bloom intermediate layer
Based on https://github.com/huggingface/transformers/commit/ca2a55e9dfb245527b5e1c954fec6ffbb7aef07b
See commit history for authorship.
"""
from typing import Optional, Tuple

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor

from petals.utils.misc import is_dummy


class WrappedBloomBlock(BloomBlock):
    def __init__(self, config):
        super().__init__(config)
        print(f"Initializing Bloom block with head_dim={self.self_attention.head_dim}, num_heads={self.num_heads}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        attention_mask: Optional[torch.Tensor] = None,
        alibi: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ):
        assert attention_mask is None, "Non-causal attention masks are not supported yet"
        batch_size, seq_length = hidden_states.shape[:2]
        if layer_past is not None and is_dummy(layer_past[0]):
            # Bloom cannot use cache if it was misconsctructed(e.g. Dummy tensors)
            # In this case, fallback to the old code:
            layer_past = None
        if layer_past is not None:
            print(f"Bloom forward received layer_past type: {type(layer_past)}")
            if isinstance(layer_past, tuple):
                print(f"Bloom cache tuple len: {len(layer_past)}")
                for i, item in enumerate(layer_past):
                    print(f"Cache item {i} type: {type(item)}, shape: {getattr(item, 'shape', 'NO SHAPE')}")
        
        try:
            # Handle combined key-value cache from other models
            if layer_past is not None:
                combined = layer_past[0]
                # Get the last dimension size
                last_dim = combined.size(-1)
                # Split the cache in half for key and value
                split_size = last_dim // 2
                past_key, past_value = torch.split(combined, [split_size, split_size], dim=-1)
                # Reconstruct the layer_past tuple in the format expected by BloomBlock
                layer_past = (past_key, past_value)
            
            past_length = 0 if layer_past is None else layer_past[0].shape[-1]
            
        except Exception as e:
            print(f"Error processing Bloom cache: {e}")
            print(f"Cache type: {type(layer_past)}, contents: {[t.shape if hasattr(t, 'shape') else type(t) for t in layer_past] if layer_past else None}")
            raise
        
        seq_length_with_past = seq_length + past_length
        attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
        if alibi is None:
            alibi = build_alibi_tensor(attention_mask, num_heads=self.self_attention.num_heads, dtype=hidden_states.dtype)
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_length,
        )
        attention_mask = attention_mask.bool()
        return super().forward(
            hidden_states,
            *args,
            attention_mask=attention_mask,
            alibi=alibi,
            layer_past=layer_past,
            **kwargs
        )
