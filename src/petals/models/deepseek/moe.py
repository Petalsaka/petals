"""MoE (Mixture of Experts) components for DeepSeek models"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.layers import register_expert_class
from hivemind.utils.tensor_descr import TensorDescriptor
from transformers.activations import ACT2FN

class DeepSeekMLP(nn.Module):
    """Base MLP module used by each expert"""
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

class Router(nn.Module):
    """Expert routing layer that computes dispatch weights for selecting experts"""
    def __init__(self, hidden_size: int, num_experts: int, capacity_factor: float = 1.25):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.router_weights = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor, *, top_k: int = 2):
        """
        Compute routing probabilities for dispatching inputs to experts
        :param hidden_states: (batch_size, seq_len, hidden_size)
        :param top_k: number of experts to route to
        :returns: scores (batch_size, seq_len, top_k), indices (batch_size, seq_len, top_k)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute routing scores
        router_logits = self.router_weights(hidden_states)  # [batch_size, seq_len, num_experts]
        routing_scores = F.softmax(router_logits, dim=-1)
        
        # Select top-k experts
        scores, indices = torch.topk(routing_scores, top_k, dim=-1)
        scores = F.softmax(scores, dim=-1)  # Normalize selected expert scores
        
        return scores, indices

class DeepSeekExpertLayer(nn.Module):
    """Expert layer that can be used with Hivemind's distributed expert computation"""
    def __init__(self, config, expert_idx: int):
        super().__init__()
        self.expert = DeepSeekMLP(config)
        self.expert_idx = expert_idx
        self.expert_uid = ExpertUID(f"expert.{expert_idx}")
        
    def forward(self, hidden_states: torch.Tensor):
        return self.expert(hidden_states)

    @staticmethod
    def get_args_schema():
        """Return the schema for expert inputs"""
        # Create a tensor descriptor with the correct size
        class CustomTensorDescriptor(TensorDescriptor):
            def make_zeros(self, batch_size=None):
                if batch_size is None:
                    batch_size = 1
                return torch.zeros(batch_size, 1, 7168, dtype=self.dtype)

        return (CustomTensorDescriptor((1, 1, 7168), dtype=torch.float32),)

    @staticmethod
    def get_kwargs_schema():
        """Return an empty kwargs schema"""
        return {}

class DeepSeekMoELayer(nn.Module):
    """Mixture of Experts layer for DeepSeek models"""
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_experts = 256  # DeepSeek-R1 specific
        self.router = Router(config.hidden_size, self.num_experts)
        self.experts = nn.ModuleList([
            DeepSeekExpertLayer(config, i) for i in range(self.num_experts)
        ])
        
    def forward(self, hidden_states: torch.Tensor):
        """
        :param hidden_states: (batch_size, seq_len, hidden_size)
        :returns: (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Get expert scores and indices
        scores, indices = self.router(hidden_states)  # scores: [batch, seq, top_k], indices: [batch, seq, top_k]
        top_k = scores.size(-1)
        
        # Reshape for expert computation
        flat_hidden = hidden_states.view(-1, hidden_size)  # [batch * seq, hidden]
        flat_scores = scores.view(-1, top_k)  # [batch * seq, top_k]
        flat_indices = indices.view(-1, top_k)  # [batch * seq, top_k]
        
        # Compute expert outputs
        expert_outputs = torch.zeros_like(flat_hidden)  # [batch * seq, hidden]
        for k in range(top_k):
            expert_idx = flat_indices[:, k]
            expert_mask = F.one_hot(expert_idx, num_classes=self.num_experts)  # [batch * seq, num_experts]
            for i, expert in enumerate(self.experts):
                mask_i = expert_mask[:, i].bool()
                if mask_i.any():
                    expert_output = expert(flat_hidden[mask_i])
                    expert_outputs[mask_i] += flat_scores[:, k][mask_i].unsqueeze(-1) * expert_output
        
        # Reshape back
        output = expert_outputs.view(batch_size, seq_len, hidden_size)
        return output 