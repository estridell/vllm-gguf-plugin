# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _VllmQwen3_5ForCausalLM,
)


class Qwen3_5ForCausalLM(_VllmQwen3_5ForCausalLM, IsHybrid):
    """Expose vLLM's shipped text-only Qwen3.5 class as a hybrid model."""
