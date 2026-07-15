# SPDX-License-Identifier: Apache-2.0

import gc
import os
from collections import Counter
from pathlib import Path

import gguf
import pytest
import torch

from vllm_gguf_plugin.quantization.utils import DEQUANT_TYPES

Q1_0 = gguf.GGMLQuantizationType.Q1_0
Q2_0 = gguf.GGMLQuantizationType.Q2_0

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_bonsai_tensor_types_are_all_recognized(ternary_model_path: Path) -> None:
    tensors = gguf.GGUFReader(ternary_model_path).tensors
    histogram = Counter(int(tensor.tensor_type) for tensor in tensors)

    assert histogram == Counter({42: 197, 0: 113})
    assert all(
        tensor.tensor_type in DEQUANT_TYPES
        or tensor.tensor_type
        in {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        }
        for tensor in tensors
    )
    token_embedding = next(t for t in tensors if t.name == "token_embd.weight")
    assert token_embedding.tensor_type == Q2_0


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Bonsai 4B memory assertion requires CUDA",
)
@pytest.mark.cuda
def test_bonsai_4b_llm_weight_load_stays_within_packed_budget(
    ternary_4b_model_path: Path,
    ternary_4b_config_path: Path,
) -> None:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.pop("VLLM_GGUF_USE_CUDA", None)
    gc.collect()
    torch.cuda.empty_cache()

    from vllm import LLM

    llm = LLM(
        model=str(ternary_4b_model_path),
        tokenizer="Qwen/Qwen3-4B",
        hf_config_path=str(ternary_4b_config_path),
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        dtype="half",
        seed=0,
    )
    try:
        runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
        expected_packed_bytes = sum(
            module.qweight.numel()
            for module in runner.model.modules()
            if hasattr(module, "qweight")
            and hasattr(module, "qweight_type")
            and int(module.qweight_type.weight_type) in (int(Q1_0), int(Q2_0))
        )
        assert expected_packed_bytes > 0
        assert runner.model_memory_usage < expected_packed_bytes * 1.15
    finally:
        llm.llm_engine.engine_core.shutdown()
