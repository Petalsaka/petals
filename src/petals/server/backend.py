from __future__ import annotations

from collections import Counter
from itertools import chain
from typing import Any, Dict, Optional, Sequence, Tuple, Type, Union, cast, TypeVar

import torch
from hivemind import BatchTensorDescriptor, TensorDescriptor
from hivemind.moe.expert_uid import ExpertUID
from hivemind.moe.server.module_backend import ModuleBackend
from hivemind.utils import get_logger
from tensor_parallel import TensorParallel
from torch import nn
from transformers import PretrainedConfig
from transformers.models.bloom.modeling_bloom import BloomAttention

from petals.data_structures import InferenceMetadata
from petals.server.memory_cache import MemoryCache
from petals.server.task_pool import PrioritizedTaskPool
from petals.utils.misc import get_size_in_bytes, is_dummy

# Type definitions
BackendType = TypeVar('BackendType', bound='TransformerBackend')

logger = get_logger(__name__)


class TransformerBackend(ModuleBackend):
    """A wrapper for a transformer block that can process requests for forward, backward and inference"""

    _peft_module = None

    def __init__(
        self,
        *args,
        config: PretrainedConfig,
        memory_cache: MemoryCache,
        backend_dtype: torch.dtype,
        max_chunk_size_bytes: int,
        **kwargs,
    ):
        import petals.utils.peft as _peft_module

        self._peft_module = _peft_module

        super().__init__(*args, **kwargs)
        assert isinstance(self.module, TensorParallel)
        self.config = config
        self.memory_cache = memory_cache
        self.max_chunk_size_bytes = max_chunk_size_bytes

        for name, param in self.module.named_parameters():
            assert not param.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"
        for name, buf in self.module.named_buffers():
            assert not buf.requires_grad, f"Block parameters must not accumulate gradients, but {name} does"

        max_batch_size = self.forward_pool.max_batch_size
        device = self.module.devices[self.module.output_device_index]
        self.inference_pool = PrioritizedTaskPool(
            self.inference_step, max_batch_size=max_batch_size, device=device, name=f"{self.name}_inference"
        )  # note: inference_pools may be merged later, see merge_inference_pools_inplace
        self.forward_pool = PrioritizedTaskPool(
            self.forward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_forward"
        )
        self.backward_pool = PrioritizedTaskPool(
            self.backward, max_batch_size=max_batch_size, device=device, name=f"{self.name}_backward"
        )

        self.dtype = backend_dtype
        self.dtype_bytes = get_size_in_bytes(self.dtype)
        self.shard_num_heads = []
        
        # Find attention modules and collect their head counts
        for shard in self.module.module_shards:
            for submodule in shard.modules():
                if hasattr(submodule, "num_heads") and isinstance(submodule, nn.Module):
                    self.shard_num_heads.append(submodule.num_heads)
                    break  # Found the attention module in this shard
                    
        assert len(self.shard_num_heads) == len(self.module.devices)
        assert sum(self.shard_num_heads) == config.num_attention_heads

        self.inference_schema = (
            (
                *self.args_schema,
                BatchTensorDescriptor((), dtype=self.dtype),
                BatchTensorDescriptor((), dtype=torch.int64),
            ),
            self.kwargs_schema,
        )

        self.cache_bytes_per_token: Dict[torch.device, int] = Counter()
        for descr in self.get_inference_cache_descriptors(batch_size=1, max_length=1):
            self.cache_bytes_per_token[descr.device] += descr.numel() * get_size_in_bytes(descr.dtype)

    def get_inference_cache_descriptors(self, batch_size: int, max_length: int) -> Sequence[TensorDescriptor]:
        """Create tensor descriptors for attention cache tensors used during inference_step"""
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        cache_tensors = []
        for device, num_heads in zip(self.module.devices, self.shard_num_heads):
            num_heads //= self.config.num_key_value_groups
            if hasattr(self.config, "num_key_value_heads"):
                num_heads = self.config.num_key_value_heads
            # Create separate key and value tensors
            keys = TensorDescriptor((batch_size, num_heads, head_dim, max_length), dtype=self.dtype, device=device)
            values = TensorDescriptor((batch_size, num_heads, max_length, head_dim), dtype=self.dtype, device=device)
            cache_tensors.extend((keys, values))
        return cache_tensors

    def forward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().forward(*inputs)

    def backward(self, *inputs: Union[torch.Tensor, str]) -> Tuple[torch.Tensor, ...]:
        *inputs, active_adapter = inputs
        with self._peft_module.using_adapter(active_adapter):
            return super().backward(*inputs)

    @torch.inference_mode()
    def inference_step(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: Optional[torch.LongTensor],
        inference_info: InferenceMetadata,
    ) -> Tuple[torch.Tensor, ...]:
        """Compute next transformer layer(s) using this server's part of the model"""
        assert hidden_states.ndim == 3, "expected hidden states to be 3-dimensional: [batch_size, seq_len, hid_size]"
        seq_len = hidden_states.shape[1]
        
        logger.debug(f"Starting inference step with hidden_states shape: {hidden_states.shape}")
        logger.debug(f"Inference info: prefix_length={inference_info.prefix_length}")
        
        if hypo_ids is None:
            logger.debug("No hypo_ids provided, creating default range")
            hypo_ids = torch.arange(hidden_states.shape[0], dtype=torch.int64, device=hidden_states.device)
        else:
            logger.debug(f"Using provided hypo_ids: {hypo_ids}")
            
        with self.memory_cache.use_cache(*inference_info.cache_handles) as cache_tensors, self._peft_module.using_adapter(inference_info.active_adapter):
            logger.debug(f"Retrieved {len(cache_tensors)} cache tensors")
            for i, tensor in enumerate(cache_tensors):
                logger.debug(f"  Cache tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            
            self._reorder_cache_inplace(cache_tensors, hypo_ids)
            
            # We chunk the inputs so that peak memory for long sequences fits into `autograd_memory`
            # reserved in `Server._choose_num_blocks()`. This saves us from OOMs if `max_chunk_size_bytes`
            # is at least 4-6x less than `autograd_memory`.
            max_chunk_length = self._estimate_max_chunk_length(hidden_states, inference_info)
            logger.debug(f"Estimated max chunk length: {max_chunk_length}")
            
            output_hidden_states = torch.empty_like(hidden_states) if seq_len > max_chunk_length else None
            layer_past = self._select_layer_past(cache_tensors, inference_info.prefix_length)
            
            logger.debug(f"Created layer_past: type={type(layer_past)}")
            if layer_past is not None:
                # Get key and value states for logging
                key_states = layer_past[0]  # Use __getitem__ to get key states
                value_states = layer_past[1]  # Use __getitem__ to get value states
                logger.debug(f"Layer past key shape: {key_states.shape}")
                logger.debug(f"Layer past value shape: {value_states.shape}")
            
            for offset in range(0, seq_len, max_chunk_length):
                chunk_length = min(max_chunk_length, seq_len - offset)
                hidden_states_chunk = hidden_states[:, offset:offset + chunk_length]
                logger.debug(f"Processing chunk at offset {offset} with length {chunk_length}")
                
                try:
                    outputs = self.module(
                        hidden_states_chunk,
                        layer_past=layer_past,
                        use_cache=True
                    )
                    logger.debug(f"Module output type: {type(outputs)}")
                    if isinstance(outputs, tuple):
                        logger.debug(f"Output tuple length: {len(outputs)}")
                        for i, out in enumerate(outputs):
                            logger.debug(f"  Output {i}: type={type(out)}, shape={getattr(out, 'shape', 'N/A')}")
                except Exception as e:
                    logger.error(f"Error in module forward pass: {str(e)}")
                    raise
                
                if isinstance(outputs, tuple):
                    output_hidden_states_chunk = outputs[0]
                    new_kvs = outputs[-1] if len(outputs) > 2 else None
                else:
                    output_hidden_states_chunk = outputs
                    new_kvs = None
                
                if seq_len > max_chunk_length:
                    output_hidden_states[:, offset:offset + chunk_length] = output_hidden_states_chunk
                else:
                    output_hidden_states = output_hidden_states_chunk
                    
                if new_kvs is not None and layer_past is not None:
                    logger.debug("Updating layer_past with new KV cache")
                    layer_past.update(*new_kvs, layer_idx=0)
            
            logger.debug(f"Completed inference step, output shape: {output_hidden_states.shape}")
            return (output_hidden_states,)

    def _estimate_max_chunk_length(self, hidden_states: torch.Tensor, inference_info: InferenceMetadata) -> int:
        # We assume that attention logit matrices are the main thing that consumes memory, given that
        # the model uses multi-query attention
        batch_size, seq_length, hidden_size = hidden_states.shape
        worst_case_length = inference_info.prefix_length + seq_length
        attn_bytes_per_token = max(self.shard_num_heads) * batch_size * self.dtype_bytes * worst_case_length
        return max(1, self.max_chunk_size_bytes // attn_bytes_per_token)

    def _reorder_cache_inplace(self, cache_tensors: Sequence[torch.Tensor], hypo_ids: torch.Tensor):
        """If hypo_ids is specified, reorder elements of each cache tensor in-place by taking indices from hypo_ids"""
        if not is_dummy(hypo_ids):
            logger.debug(f"Reordering cache tensors with hypo_ids: {hypo_ids}")
            for i, cache_tensor in enumerate(cache_tensors):
                logger.debug(f"  Before reorder - Cache tensor {i}: shape={cache_tensor.shape}, dtype={cache_tensor.dtype}")
                try:
                    cache_tensor[...] = cache_tensor[hypo_ids.to(cache_tensor.device)]  # in-place reorder cache by hypo ids
                    logger.debug(f"  After reorder - Cache tensor {i}: shape={cache_tensor.shape}, dtype={cache_tensor.dtype}")
                except Exception as e:
                    logger.error(f"Error reordering cache tensor {i}: {str(e)}")
                    logger.error(f"  Tensor shape: {cache_tensor.shape}")
                    logger.error(f"  Hypo_ids shape: {hypo_ids.shape}")
                    logger.error(f"  Hypo_ids device: {hypo_ids.device}")
                    logger.error(f"  Tensor device: {cache_tensor.device}")
                    raise

    def _select_layer_past(self, cache_tensors: Sequence[torch.Tensor], prefix_length: int) -> Optional[_BloomCacheWrapper]:
        if not cache_tensors:
            return None

        if len(cache_tensors) % 2 != 0:
            raise ValueError(f"Expected even number of cache tensors (key-value pairs), got {len(cache_tensors)}")
            
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        
        # Process key and value tensors separately
        key_tensors = []
        value_tensors = []
        
        for i in range(0, len(cache_tensors), 2):
            key = cache_tensors[i]
            value = cache_tensors[i + 1]
            
            batch_size, num_heads = key.shape[:2]
            
            # Process key: [batch, heads, head_dim, seq_len] -> [batch*heads, seq_len, head_dim]
            key = key.view(batch_size, num_heads, head_dim, -1)
            key = key.permute(0, 1, 3, 2)  # [batch, heads, seq_len, head_dim]
            key = key.reshape(batch_size * num_heads, -1, head_dim)
            
            # Process value: [batch, heads, seq_len, head_dim] -> [batch*heads, seq_len, head_dim]
            value = value.view(batch_size, num_heads, -1, head_dim)
            value = value.reshape(batch_size * num_heads, -1, head_dim)
            
            key_tensors.append(key)
            value_tensors.append(value)
        
        # Concatenate along the batch*heads dimension
        combined_key = torch.cat(key_tensors, dim=0)  # [total_heads, seq_len, head_dim]
        combined_value = torch.cat(value_tensors, dim=0)  # [total_heads, seq_len, head_dim]
        
        # Create a combined tensor that can be split by the Bloom block
        # The block expects to split this into [head_dim, head_dim] along dim=-1
        combined = torch.cat([combined_key, combined_value], dim=-1)  # [total_heads, seq_len, head_dim*2]
        
        return _BloomCacheWrapper(combined)

    def _update_cache_inplace(
        self, cache_tensors: Sequence[torch.Tensor], new_kvs: Sequence[torch.Tensor], prefix_length: int
    ):
        """Update cache tensors in-place with new key-value pairs."""
        if len(cache_tensors) != len(new_kvs):
            raise ValueError(f"Number of cache tensors ({len(cache_tensors)}) must match number of new tensors ({len(new_kvs)})")
            
        if len(cache_tensors) % 2 != 0:
            raise ValueError(f"Expected even number of cache tensors (key-value pairs), got {len(cache_tensors)}")
            
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        new_length = prefix_length + new_kvs[0].shape[1]  # prefix_length + new sequence length
        
        logger.debug(f"Updating cache tensors with prefix_length={prefix_length}, new_length={new_length}")
        
        # Process pairs of tensors (key, value)
        for i in range(0, len(cache_tensors), 2):
            cache_key = cache_tensors[i]
            cache_value = cache_tensors[i + 1]
            new_key = new_kvs[i]
            new_value = new_kvs[i + 1]
            
            logger.debug(f"Processing cache pair {i//2}:")
            logger.debug(f"  Cache key shape: {cache_key.shape}, new key shape: {new_key.shape}")
            logger.debug(f"  Cache value shape: {cache_value.shape}, new value shape: {new_value.shape}")
            
            batch_size, num_heads = cache_key.shape[:2]
            
            # For Bloom:
            # - key should be [batch, heads, head_dim, seq_len]
            # - value should be [batch, heads, seq_len, head_dim]
            
            # Reshape new key tensor
            new_key = new_key.view(batch_size * num_heads, -1, head_dim)  # [batch*heads, seq_len, head_dim]
            new_key = new_key.permute(0, 2, 1)  # [batch*heads, head_dim, seq_len]
            new_key = new_key.reshape(batch_size, num_heads, head_dim, -1)  # [batch, heads, head_dim, seq_len]
            
            # Reshape new value tensor
            new_value = new_value.view(batch_size * num_heads, -1, head_dim)  # [batch*heads, seq_len, head_dim]
            new_value = new_value.reshape(batch_size, num_heads, -1, head_dim)  # [batch, heads, seq_len, head_dim]
            
            # Update the cache tensors in-place
            cache_key[:, :, :, prefix_length:new_length] = new_key[:, :, :, prefix_length:new_length]
            cache_value[:, :, prefix_length:new_length, :] = new_value[:, :, prefix_length:new_length, :]

    def get_pools(self) -> Sequence[PrioritizedTaskPool]:
        return self.forward_pool, self.backward_pool, self.inference_pool

    def get_info(self) -> Dict[str, Any]:
        """Get module parameters and stats. Used by RemoteExpert to check shapes and for DMoE orchestration."""
        return dict(super().get_info(), inference_schema=self.inference_schema)

    def shutdown(self):
        # Break the cyclic references, otherwise TransformerBackend may be not garbage-collected
        self.forward_pool = self.backward_pool = self.inference_pool = None

        # Explicitly free the GPU memory. This is not necessary at the time this code is written,
        # but may help to avoid future issues when the module is not garbage-collected for some reasons
        dummy = torch.tensor([])
        for p in self.module.parameters():
            p.data = dummy


def merge_inference_pools_inplace(backends: Dict[str, 'TransformerBackend']) -> None:
    """Replace each backend's rpc_inference pools with a combined pool runs multiple blocks in one call"""
    assert len(backends) != 0 and all(isinstance(b, TransformerBackend) for b in backends.values())
    first_pool = next(iter(backends.values())).inference_pool
    merged_pool = PrioritizedTaskPool(
        _MergedInferenceStep(backends),
        max_batch_size=first_pool.max_batch_size,
        device=first_pool.device,
        name=f"merged_inference",
    )
    for backend in backends.values():
        assert not backend.inference_pool.is_alive()
        backend.inference_pool = merged_pool


class _MergedInferenceStep:
    def __init__(self, backends: Dict[str, 'TransformerBackend']) -> None:
        self.backends = backends

    @torch.inference_mode()
    def __call__(
        self,
        hidden_states: torch.Tensor,
        hypo_ids: torch.LongTensor,
        inference_infos: Sequence[InferenceMetadata],
        *optional_prompts: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, ...]:
        assert len(inference_infos) == len(
            optional_prompts
        ), f"found {len(inference_infos)} blocks but {len(optional_prompts)} prompts"
        for inference_info, optional_prompt in zip(inference_infos, optional_prompts):
            if optional_prompt is not None:
                hidden_states[:, : optional_prompt.shape[1]] += optional_prompt
            (hidden_states,) = self.backends[inference_info.uid].inference_step(hidden_states, hypo_ids, inference_info)
        return (hidden_states,)


# Add the Bloom cache wrapper class to handle position tracking
class _BloomCacheWrapper:
    def __init__(self, combined: torch.Tensor):
        assert isinstance(combined, torch.Tensor), "Cache wrapper requires tensor input"
        
        # Validate tensor shapes
        if combined.ndim != 3:
            raise ValueError(f"Expected 3D tensors [total_heads, seq_len, head_dim*2], got {combined.ndim}")
            
        logger.debug(f"Creating _BloomCacheWrapper with shapes:")
        logger.debug(f"  Combined: {combined.shape}")
        
        self.combined = combined
        
    def __getitem__(self, index: int):
        logger.debug(f"Accessing cache wrapper with index {index}")
        if not isinstance(index, int):
            raise TypeError(f"Cache wrapper index must be integer, got {type(index)}")
        if index not in (0, 1):
            raise IndexError(f"Cache wrapper index must be 0 or 1, got {index}")
            
        if index == 0:
            return self.combined[:, :, :self.combined.shape[2]//2]
        else:
            return self.combined[:, :, self.combined.shape[2]//2:]

    def __iter__(self):
        logger.debug("Iterating over cache wrapper")
        yield self.combined[:, :, :self.combined.shape[2]//2]
        yield self.combined[:, :, self.combined.shape[2]//2:]

    def update(self, new_key: torch.Tensor, new_value: torch.Tensor, layer_idx: int = 0):
        logger.debug(f"Updating cache with shapes:")
        logger.debug(f"  New key: {new_key.shape}")
        logger.debug(f"  New value: {new_value.shape}")
        
        # Validate new tensors
        if new_key.ndim != 3 or new_value.ndim != 3:
            raise ValueError(f"Expected 3D tensors [total_heads, seq_len, head_dim], got key.shape={new_key.shape}, value.shape={new_value.shape}")
            
        if new_key.shape != new_value.shape:
            raise ValueError(f"Key and value shapes must match, got key={new_key.shape}, value={new_value.shape}")
            
        # Update the cache tensors
        self.combined = torch.cat([new_key, new_value], dim=-1)

    @property
    def shape(self):
        return self.combined.shape
