# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import gguf
import torch
from transformers import Qwen3_5Config, Qwen3_5TextConfig
from vllm.model_executor.models.registry import ModelRegistry

import vllm_gguf_plugin.models.qwen3_5 as qwen35_model_module
from vllm_gguf_plugin import register
from vllm_gguf_plugin.config_parser import QWEN35_GGUF_ARCHITECTURE
from vllm_gguf_plugin.quantization.config import GGUFConfig
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


def test_qwen35_composite_config_uses_text_model_for_name_map() -> None:
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForCausalLM"],
        text_config=_config(),
    )
    adapter = get_weights_adapter(config)
    assert isinstance(adapter, Qwen35GGUFAdapter)

    model_config = SimpleNamespace(hf_config=config, trust_remote_code=False)
    name_map = adapter.build_name_map(model_config)

    assert len(name_map) == 56
    assert len(set(name_map.values())) == 56
    assert all("vision" not in name for name in name_map.values())


def test_qwen35_and_qwen36_non_gguf_registration_is_unchanged() -> None:
    # Both families use the canonical Qwen3_5ForCausalLM architecture.
    original_entry = ModelRegistry.models.get("Qwen3_5ForCausalLM")

    register()

    assert ModelRegistry.models.get("Qwen3_5ForCausalLM") is original_entry


def test_qwen35_gguf_model_is_registered_as_hybrid() -> None:
    register()
    entry = ModelRegistry.models[QWEN35_GGUF_ARCHITECTURE]
    info = entry.inspect_model_cls()
    model_class = entry.load_model_cls()

    assert info.is_text_generation_model
    assert info.is_hybrid
    assert model_class.__module__ == "vllm_gguf_plugin.models.qwen3_5"
    assert hasattr(model_class, "get_mamba_state_dtype_from_config")
    assert hasattr(model_class, "get_mamba_state_shape_from_config")
    assert hasattr(model_class, "get_mamba_state_copy_func")


def test_qwen35_gguf_model_replaces_unquantized_embedding(monkeypatch) -> None:
    calls = []

    def fake_base_init(self, *, vllm_config, prefix=""):
        torch.nn.Module.__init__(self)
        self.quant_config = vllm_config.quant_config
        self.config = vllm_config.model_config.hf_text_config
        self.model = SimpleNamespace(embed_tokens=object())
        self.lm_head = object()

    def fake_embedding(*args, **kwargs):
        calls.append((args, kwargs))
        return "packed-embedding"

    monkeypatch.setattr(
        qwen35_model_module._VllmQwen3_5ForCausalLM, "__init__", fake_base_init
    )
    monkeypatch.setattr(qwen35_model_module, "VocabParallelEmbedding", fake_embedding)
    config = SimpleNamespace(
        vocab_size=248320,
        hidden_size=5120,
        tie_word_embeddings=False,
    )
    quant_config = GGUFConfig()
    quant_config.ternary_modules.add("language_model.model.embed_tokens")
    vllm_config = SimpleNamespace(
        quant_config=quant_config,
        model_config=SimpleNamespace(hf_text_config=config),
    )

    model = qwen35_model_module.Qwen3_5ForCausalLM(
        vllm_config=vllm_config, prefix="language_model"
    )

    assert model.model.embed_tokens == "packed-embedding"
    assert calls == [
        (
            (248320, 5120),
            {
                "quant_config": vllm_config.quant_config,
                "prefix": "language_model.model.embed_tokens",
            },
        )
    ]

    kquant_config = GGUFConfig()
    kquant_model = qwen35_model_module.Qwen3_5ForCausalLM(
        vllm_config=SimpleNamespace(
            quant_config=kquant_config,
            model_config=SimpleNamespace(hf_text_config=config),
        )
    )
    assert kquant_model.model.embed_tokens != "packed-embedding"
    assert len(calls) == 1


def test_qwen35_load_spec_marks_only_ternary_modules(monkeypatch) -> None:
    adapter = Qwen35GGUFAdapter(_config())
    monkeypatch.setattr(adapter, "patch_hf_config", lambda path, config: config)
    monkeypatch.setattr(adapter, "build_name_map", lambda config: {})
    monkeypatch.setattr(adapter, "update_tie_word_embeddings", lambda *args: None)
    monkeypatch.setattr(
        adapter,
        "get_weight_type_map",
        lambda *args: {
            "model.embed_tokens.weight": "Q2_0",
            "lm_head.weight": "Q4_K",
            "model.norm.weight": "F32",
        },
    )
    model_config = SimpleNamespace(hf_config=_config())

    load_spec = adapter.prepare_loading("/tmp/model.gguf", model_config)

    assert load_spec.ternary_modules == {"model.embed_tokens"}
    assert load_spec.unquantized_modules == ["model.norm"]


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
