"""Wrapper for DeepSeek block that maps parameter names to match the Hugging Face implementation"""

import torch
import torch.nn as nn
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.moe.server.runtime import Runtime
from hivemind.utils.tensor_descr import TensorDescriptor
from petals.models.deepseek.config import DistributedDeepSeekConfig

class WrappedDeepSeekBlock(nn.Module):
    """A wrapper around DeepSeek block that maps parameter names to match the Hugging Face implementation"""

    def __init__(self, config: DistributedDeepSeekConfig, layer_idx: int = None):
        super().__init__()
        if layer_idx is None:
            raise ValueError("layer_idx must be provided")
        
        # Lazy import to avoid circular dependency
        from petals.models.deepseek.block import DeepSeekBlock
        self.block = DeepSeekBlock(config, layer_idx)
        self.layer_idx = layer_idx

        # Initialize module backends for expert computation
        self.module_backends = {}
        for i in range(self.block.moe.num_experts):
            expert = self.block.moe.experts[i]
            expert_name = f"expert.{i}"
            
            # Get schemas from expert
            args_schema = expert.get_args_schema()
            kwargs_schema = expert.get_kwargs_schema()
            
            self.module_backends[expert_name] = ModuleBackend(
                name=expert_name,
                module=expert,
                args_schema=args_schema,
                kwargs_schema=kwargs_schema,
                max_batch_size=2048,  # Large enough for typical sequence lengths
            )

        # Initialize runtime with module backends
        self.runtime = Runtime(module_backends=self.module_backends)

        # Map from HuggingFace parameter names to internal names
        self._param_mapping = {
            # Attention parameters
            f'model.layers.{self.layer_idx}.self_attn.q_a_proj.weight': 'block.self_attn.q_a_proj.weight',
            f'model.layers.{self.layer_idx}.self_attn.q_b_proj.weight': 'block.self_attn.q_b_proj.weight',
            f'model.layers.{self.layer_idx}.self_attn.kv_a_proj_with_mqa.weight': 'block.self_attn.kv_a_proj_with_mqa.weight',
            f'model.layers.{self.layer_idx}.self_attn.kv_b_proj.weight': 'block.self_attn.kv_b_proj.weight',
            f'model.layers.{self.layer_idx}.self_attn.q_a_layernorm.weight': 'block.self_attn.q_a_layernorm.weight',
            f'model.layers.{self.layer_idx}.self_attn.kv_a_layernorm.weight': 'block.self_attn.kv_a_layernorm.weight',
            f'model.layers.{self.layer_idx}.self_attn.o_proj.weight': 'block.self_attn.o_proj.weight',
            
            # Layer norms
            f'model.layers.{self.layer_idx}.input_layernorm.weight': 'block.input_layernorm.weight',
            f'model.layers.{self.layer_idx}.post_attention_layernorm.weight': 'block.post_attention_layernorm.weight',
            
            # Router parameters
            f'model.layers.{self.layer_idx}.moe.router.router_weights.weight': 'block.moe.router.router_weights.weight',
        }

        # Add expert parameters
        for i in range(256):  # DeepSeek-R1 specific number of experts
            expert_params = {
                f'model.layers.{self.layer_idx}.moe.experts.{i}.expert.gate_proj.weight': f'block.moe.experts.{i}.expert.gate_proj.weight',
                f'model.layers.{self.layer_idx}.moe.experts.{i}.expert.up_proj.weight': f'block.moe.experts.{i}.expert.up_proj.weight',
                f'model.layers.{self.layer_idx}.moe.experts.{i}.expert.down_proj.weight': f'block.moe.experts.{i}.expert.down_proj.weight',
            }
            self._param_mapping.update(expert_params)

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