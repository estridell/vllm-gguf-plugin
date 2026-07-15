# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as _VllmQwen3_5ForConditionalGeneration,
    Qwen3_5ForCausalLM as _VllmQwen3_5ForCausalLM,
)


class Qwen3_5ForCausalLM(_VllmQwen3_5ForCausalLM, IsHybrid):
    """Expose vLLM's shipped text-only Qwen3.5 class as a hybrid model."""

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        del cls
        return _VllmQwen3_5ForConditionalGeneration.get_mamba_state_dtype_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        del cls
        return _VllmQwen3_5ForConditionalGeneration.get_mamba_state_shape_from_config(
            vllm_config
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        del cls
        return _VllmQwen3_5ForConditionalGeneration.get_mamba_state_copy_func()
