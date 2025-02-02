from petals.utils.auto_config import register_model_classes
from petals.models.deepseek.config import DistributedDeepSeekConfig
from petals.models.deepseek.model import DistributedDeepSeekModel, DistributedDeepSeekForCausalLM
from petals.models.deepseek.wrapped_block import WrappedDeepSeekBlock

register_model_classes(
    config=DistributedDeepSeekConfig,
    model=DistributedDeepSeekModel,
    model_for_causal_lm=DistributedDeepSeekForCausalLM,
) 