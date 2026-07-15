# SPDX-License-Identifier: Apache-2.0

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.interfaces import IsHybrid
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForCausalLM as _VllmQwen3_5ForCausalLM,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration as _VllmQwen3_5ForConditionalGeneration,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..quantization.config import GGUFConfig


class Qwen3_5ForCausalLM(_VllmQwen3_5ForCausalLM, IsHybrid):
    """Expose vLLM's shipped text-only Qwen3.5 class as a hybrid model."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # vLLM's Qwen3_5Model currently omits quant_config when constructing
        # embed_tokens.  Replace only the GGUF instance before weights load so
        # token_embd remains packed; ParallelLMHead already receives it.
        embedding_name = "model.embed_tokens"
        embedding_prefix = maybe_prefix(prefix, embedding_name)
        if isinstance(self.quant_config, GGUFConfig) and (
            embedding_name in self.quant_config.ternary_modules
            or embedding_prefix in self.quant_config.ternary_modules
        ):
            embed_tokens = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=self.quant_config,
                prefix=embedding_prefix,
            )
            self.model.embed_tokens = embed_tokens
            if self.config.tie_word_embeddings:
                self.lm_head = embed_tokens

    # Qwen3.5/3.6 use M-RoPE position encoding (rope dimension sections) even
    # for text-only inputs; vLLM asserts supports_mrope on the model class.
    supports_mrope = True

    def get_mrope_input_positions(self, input_tokens, mm_features):
        if mm_features:
            raise ValueError(
                "This text-only GGUF Qwen3.5/3.6 model cannot take multimodal inputs"
            )
        num_tokens = len(input_tokens)
        llm_positions = torch.arange(num_tokens).unsqueeze(0).expand(3, -1)
        return llm_positions, 0

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
