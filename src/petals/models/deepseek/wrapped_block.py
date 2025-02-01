"""Wrapper for DeepSeek block that maps parameter names to match the Hugging Face implementation"""

import torch.nn as nn
from petals.models.deepseek.config import DistributedDeepSeekConfig

class WrappedDeepSeekBlock(nn.Module):
    """A wrapper around DeepSeek block that maps parameter names to match the Hugging Face implementation"""

    def __init__(self, config: DistributedDeepSeekConfig, layer_idx: int = None):
        super().__init__()
        # Lazy import to avoid circular dependency
        from petals.models.deepseek.block import DeepSeekBlock
        self.block = DeepSeekBlock(config, layer_idx)
        self.layer_idx = layer_idx

        # Map from HuggingFace parameter names to internal names
        self._param_mapping = {
            # Attention parameters
            f'model.layers.{layer_idx}.self_attn.q_a_proj.weight': 'block.self_attn.q_a_proj.weight',
            f'model.layers.{layer_idx}.self_attn.q_b_proj.weight': 'block.self_attn.q_b_proj.weight',
            f'model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight': 'block.self_attn.kv_a_proj_with_mqa.weight',
            f'model.layers.{layer_idx}.self_attn.kv_b_proj.weight': 'block.self_attn.kv_b_proj.weight',
            f'model.layers.{layer_idx}.self_attn.q_a_layernorm.weight': 'block.self_attn.q_a_layernorm.weight',
            f'model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight': 'block.self_attn.kv_a_layernorm.weight',
            f'model.layers.{layer_idx}.self_attn.o_proj.weight': 'block.self_attn.o_proj.weight',
            
            # Layer norms
            f'model.layers.{layer_idx}.input_layernorm.weight': 'block.input_layernorm.weight',
            f'model.layers.{layer_idx}.post_attention_layernorm.weight': 'block.post_attention_layernorm.weight',
            
            # MLP parameters
            f'model.layers.{layer_idx}.mlp.gate_proj.weight': 'block.mlp.gate_proj.weight',
            f'model.layers.{layer_idx}.mlp.up_proj.weight': 'block.mlp.up_proj.weight',
            f'model.layers.{layer_idx}.mlp.down_proj.weight': 'block.mlp.down_proj.weight',
        }

        # Create reverse mapping for state_dict conversion
        self._reverse_param_mapping = {v: k for k, v in self._param_mapping.items()}

    def state_dict(self, *args, **kwargs):
        """Convert internal parameter names to match the Hugging Face implementation"""
        state_dict = self.block.state_dict(*args, **kwargs)
        mapped_state_dict = {}
        for name, param in state_dict.items():
            # Find the HuggingFace name for this internal parameter
            if name in self._reverse_param_mapping:
                mapped_state_dict[self._reverse_param_mapping[name]] = param
            else:
                mapped_state_dict[name] = param
        return mapped_state_dict

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Convert Hugging Face parameter names to internal names"""
        mapped_state_dict = {}
        
        # First, try to find parameters with the full prefix
        for external, internal in self._param_mapping.items():
            full_external = prefix + external
            full_internal = prefix + internal
            
            # Try different possible locations of the parameter
            if full_external in state_dict:
                mapped_state_dict[full_internal] = state_dict[full_external]
            elif external in state_dict:
                mapped_state_dict[full_internal] = state_dict[external]
            elif f"model.{external}" in state_dict:
                mapped_state_dict[full_internal] = state_dict[f"model.{external}"]
            
        # Include any remaining parameters
        for key, value in state_dict.items():
            if not any(internal in key for internal in self._param_mapping.values()):
                mapped_state_dict[key] = value
                
        return super()._load_from_state_dict(mapped_state_dict, prefix, *args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs) 