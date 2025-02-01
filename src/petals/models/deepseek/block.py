"""
DeepSeek-R1 intermediate layer
Based on DeepSeek-V3 architecture
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from petals.utils.cuda_graphs import make_inference_graphed_callable


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepSeekRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class DeepSeekRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=131072, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.register_buffer("inv_freq", None, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if self.inv_freq is None or self.inv_freq.device != x.device:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        freqs = position_ids.float() * self.inv_freq.view(1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin


class DeepSeekAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.max_position_embeddings = config.max_position_embeddings

        # Split query projection into a and b parts
        self.q_a_layernorm = nn.LayerNorm(1536, bias=False)
        self.q_a_proj = nn.Linear(self.hidden_size, 1536, bias=config.attention_bias)
        self.q_b_proj = nn.Linear(1536, 24576, bias=config.attention_bias)
        
        # Split key/value projection into a and b parts with MQA
        self.kv_a_layernorm = nn.LayerNorm(512, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, 576, bias=config.attention_bias)
        self.kv_b_proj = nn.Linear(512, 32768, bias=config.attention_bias)
        
        self.o_proj = nn.Linear(16384, self.hidden_size, bias=config.attention_bias)

        self.rotary_emb = DeepSeekRotaryEmbedding(
            self.qk_rope_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        # Process query states with a/b projections
        q_a_states = self.q_a_layernorm(hidden_states)
        q_a_states = self.q_a_proj(q_a_states)
        q_b_states = self.q_b_proj(q_a_states)
        q_a_states = q_a_states.view(bsz, q_len, 128, 12)  # 1536 = 128 * 12
        q_b_states = q_b_states.view(bsz, q_len, 128, 192)  # 24576 = 128 * 192
        
        # Process key/value states with a/b projections and MQA
        kv_a_states = self.kv_a_layernorm(hidden_states)
        kv_a_states = self.kv_a_proj_with_mqa(kv_a_states)  # [batch, seq, 576]
        k_states = kv_a_states[:, :, :512].view(bsz, q_len, 128, 4)  # 512 = 128 * 4
        v_a_states = kv_a_states[:, :, 512:].view(bsz, q_len, 128, 0.5)  # 64 = 128 * 0.5
        
        kv_b_states = self.kv_b_proj(k_states.view(bsz, q_len, 512))  # [batch, seq, 32768]
        v_b_states = kv_b_states.view(bsz, q_len, 128, 256)  # 32768 = 128 * 256
        
        value_states = torch.cat([v_a_states, v_b_states], dim=-1)

        q_a_states = q_a_states.transpose(1, 2)
        q_b_states = q_b_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos, sin = self.rotary_emb(k_states, position_ids)
        q_a_states, k_states = apply_rotary_pos_emb(q_a_states, k_states, cos, sin)

        if past_key_value is not None:
            k_states = torch.cat([past_key_value[0], k_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (k_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        k_states = k_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=0)
        value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=0)

        # Compute attention scores separately for rotary and non-rotary parts
        attn_weights_a = torch.matmul(q_a_states, k_states.transpose(2, 3)) / math.sqrt(12)  # 12 is head_dim for q_a
        attn_weights_b = torch.matmul(q_b_states, k_states.transpose(2, 3)) / math.sqrt(192)  # 192 is head_dim for q_b
        attn_weights = torch.cat([attn_weights_a, attn_weights_b], dim=1)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_a_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, 16384)  # 16384 = 128 * 128

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DeepSeekMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        self.register_buffer('gate_proj_weight_scale_inv', torch.ones(1))
        self.register_buffer('up_proj_weight_scale_inv', torch.ones(1))
        self.register_buffer('down_proj_weight_scale_inv', torch.ones(1))

    def forward(self, x):
        gate = self.gate_proj(x) * self.gate_proj_weight_scale_inv
        up = self.up_proj(x) * self.up_proj_weight_scale_inv
        down = self.down_proj(self.act_fn(gate) * up) * self.down_proj_weight_scale_inv
        return down


class DeepSeekBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DeepSeekAttention(config)
        self.mlp = DeepSeekMLP(config)
        self.input_layernorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepSeekRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs 