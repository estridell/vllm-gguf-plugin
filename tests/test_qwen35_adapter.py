# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import gguf
import torch
from transformers import Qwen3_5TextConfig
from vllm.model_executor.models.registry import ModelRegistry

from vllm_gguf_plugin import register
from vllm_gguf_plugin.weights_adapter import (
    Qwen35GGUFAdapter,
    get_weights_adapter,
)


def _config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        layer_types=[
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ],
    )


def test_qwen35_name_map_covers_meta_model() -> None:
    config = _config()
    adapter = get_weights_adapter(config)
    assert isinstance(adapter, Qwen35GGUFAdapter)

    model_config = SimpleNamespace(hf_config=config, trust_remote_code=False)
    name_map = adapter.build_name_map(model_config)

    assert len(name_map) == 56
    assert len(set(name_map.values())) == 56
    assert name_map["blk.0.ssm_a"] == "model.layers.0.linear_attn.A_log"
    assert name_map["blk.0.ssm_dt.bias"] == "model.layers.0.linear_attn.dt_bias"
    assert name_map["blk.0.attn_qkv.weight"].endswith("in_proj_qkv.weight")
    assert name_map["blk.3.attn_q.weight"].endswith("self_attn.q_proj.weight")


def test_qwen35_text_model_is_registered_as_hybrid() -> None:
    register()
    entry = ModelRegistry.models["Qwen3_5ForCausalLM"]
    info = entry.inspect_model_cls()
    model_class = entry.load_model_cls()

    assert info.is_text_generation_model
    assert info.is_hybrid
    assert model_class.__module__ == "vllm_gguf_plugin.models.qwen3_5"


def test_qwen35_undoes_tiled_v_head_rows() -> None:
    adapter = Qwen35GGUFAdapter(_config())
    grouped = torch.arange(6 * 128).reshape(6 * 128, 1)
    tiled = (
        grouped.reshape(2, 3, 128, 1).transpose(0, 1).contiguous().reshape_as(grouped)
    )
    assert torch.equal(adapter._undo_v_head_reorder(tiled, 0, 128), grouped)


def test_qwen35_undoes_q2_0_tiled_out_projection_columns() -> None:
    adapter = Qwen35GGUFAdapter(_config())
    module = "model.layers.0.linear_attn.out_proj"
    adapter.transform_weight(
        module + ".qweight_type",
        torch.tensor(gguf.GGMLQuantizationType.Q2_0),
    )
    _, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q2_0]
    grouped = torch.arange(4 * 6 * type_size, dtype=torch.int64).reshape(
        4, 6 * type_size
    )
    tiled = (
        grouped.reshape(4, 2, 3, type_size)
        .transpose(1, 2)
        .contiguous()
        .reshape_as(grouped)
    )
    restored = adapter.transform_weight(module + ".qweight", tiled)
    assert torch.equal(restored, grouped)


def test_qwen35_restores_a_log_conv_shape_and_zero_centered_norm() -> None:
    adapter = Qwen35GGUFAdapter(_config())
    grouped_a = torch.arange(6, dtype=torch.float32)
    tiled_a = grouped_a.reshape(2, 3).transpose(0, 1).reshape(6)
    gguf_a = -torch.exp(tiled_a)
    restored_a = adapter.transform_weight("model.layers.0.linear_attn.A_log", gguf_a)
    assert torch.equal(restored_a, grouped_a)

    grouped_conv = torch.arange(512 + 6 * 128, dtype=torch.float32).reshape(-1, 1)
    qk, v = grouped_conv[:512], grouped_conv[512:]
    tiled_v = v.reshape(2, 3, 128, 1).transpose(0, 1).reshape_as(v)
    restored_conv = adapter.transform_weight(
        "model.layers.0.linear_attn.conv1d.weight",
        torch.cat((qk, tiled_v)).repeat(1, 4),
    )
    assert restored_conv.shape == (1280, 1, 4)

    norm = adapter.transform_weight(
        "model.layers.0.input_layernorm.weight", torch.ones(8)
    )
    linear_norm = adapter.transform_weight(
        "model.layers.0.linear_attn.norm.weight", torch.ones(8)
    )
    assert torch.equal(norm, torch.zeros(8))
    assert torch.equal(linear_norm, torch.ones(8))
