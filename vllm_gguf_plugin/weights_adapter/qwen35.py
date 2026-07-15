# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gguf
import torch

from .default import GGUFWeightsAdapter


class Qwen35GGUFAdapter(GGUFWeightsAdapter):
    """Adapter for text-only Qwen3.5/3.6 GGUF checkpoints."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self._quant_types: dict[str, gguf.GGMLQuantizationType] = {}

    @classmethod
    def matches(cls, config) -> bool:
        if config.model_type in ("qwen3_5", "qwen3_5_text"):
            return True
        text_config = getattr(config, "get_text_config", lambda: config)()
        return getattr(text_config, "model_type", None) == "qwen3_5_text"

    def get_gguf_model_type(self, config) -> str:
        del config
        return "qwen35"

    def get_explicit_name_map(self, config) -> dict[str, str]:
        text_config = config.get_text_config()
        result: dict[str, str] = {}
        for index, layer_type in enumerate(text_config.layer_types):
            if layer_type != "linear_attention":
                continue
            prefix = f"model.layers.{index}.linear_attn"
            result[f"blk.{index}.ssm_a"] = f"{prefix}.A_log"
            result[f"blk.{index}.ssm_dt.bias"] = f"{prefix}.dt_bias"
        return result

    @property
    def _num_k_heads(self) -> int:
        return self.config.get_text_config().linear_num_key_heads

    @property
    def _num_v_heads(self) -> int:
        return self.config.get_text_config().linear_num_value_heads

    def _undo_v_head_reorder(
        self,
        tensor: torch.Tensor,
        dim: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Convert Prism's tiled GGUF V-head order back to HF grouped order."""
        shape = list(tensor.shape)
        if dim < 0:
            dim += len(shape)
        num_v_per_k = self._num_v_heads // self._num_k_heads
        expected = self._num_v_heads * head_dim
        if shape[dim] != expected:
            raise ValueError(
                f"Expected V dimension {expected} at dim {dim}, got {shape[dim]}"
            )
        tiled_shape = (
            shape[:dim] + [num_v_per_k, self._num_k_heads, head_dim] + shape[dim + 1 :]
        )
        tensor = tensor.reshape(tiled_shape)
        permutation = list(range(len(tiled_shape)))
        permutation[dim], permutation[dim + 1] = (
            permutation[dim + 1],
            permutation[dim],
        )
        return tensor.permute(permutation).contiguous().reshape(shape)

    def _undo_qkv_reorder(self, weight: torch.Tensor) -> torch.Tensor:
        config = self.config.get_text_config()
        qk_dim = config.linear_num_key_heads * config.linear_key_head_dim
        q, k, v = torch.split(weight, [qk_dim, qk_dim, weight.shape[0] - 2 * qk_dim])
        v = self._undo_v_head_reorder(v, 0, config.linear_value_head_dim)
        return torch.cat((q, k, v), dim=0)

    def _undo_conv_reorder(self, weight: torch.Tensor) -> torch.Tensor:
        config = self.config.get_text_config()
        qk_channels = 2 * config.linear_num_key_heads * config.linear_key_head_dim
        qk, v = torch.split(weight, [qk_channels, weight.shape[0] - qk_channels])
        v = self._undo_v_head_reorder(v, 0, config.linear_value_head_dim)
        return torch.cat((qk, v), dim=0)

    def _quantized_head_size(self, module_name: str) -> int:
        quant_type = self._quant_types[module_name]
        block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
        head_dim = self.config.get_text_config().linear_value_head_dim
        if head_dim % block_size:
            raise ValueError(
                f"Cannot reorder {quant_type.name} columns with head_dim={head_dim} "
                f"and quantization block_size={block_size}"
            )
        return head_dim // block_size * type_size

    def transform_weight(self, hf_name: str, weight: torch.Tensor) -> torch.Tensor:
        if hf_name.endswith(".qweight_type"):
            module_name = hf_name.removesuffix(".qweight_type")
            self._quant_types[module_name] = gguf.GGMLQuantizationType(
                int(weight.item())
            )
            return weight

        is_qweight = hf_name.endswith(".qweight")
        parameter_name = (
            hf_name.removesuffix(".qweight") + ".weight" if is_qweight else hf_name
        )

        if parameter_name.endswith("norm.weight") and not parameter_name.endswith(
            "linear_attn.norm.weight"
        ):
            return weight - 1

        if ".linear_attn." not in parameter_name:
            return weight

        if parameter_name.endswith(".in_proj_qkv.weight"):
            return self._undo_qkv_reorder(weight)
        if parameter_name.endswith(".in_proj_z.weight"):
            return self._undo_v_head_reorder(
                weight,
                0,
                self.config.get_text_config().linear_value_head_dim,
            )
        if parameter_name.endswith((".in_proj_a.weight", ".in_proj_b.weight")):
            return self._undo_v_head_reorder(weight, 0, 1)
        if parameter_name.endswith(".out_proj.weight"):
            head_size = (
                self._quantized_head_size(hf_name.removesuffix(".qweight"))
                if is_qweight
                else self.config.get_text_config().linear_value_head_dim
            )
            return self._undo_v_head_reorder(weight, 1, head_size)
        if parameter_name.endswith(".conv1d.weight"):
            weight = self._undo_conv_reorder(weight)
            return weight.unsqueeze(1) if weight.ndim == 2 else weight
        if parameter_name.endswith(".A_log"):
            weight = torch.log(-weight)
            return self._undo_v_head_reorder(weight, 0, 1)
        if parameter_name.endswith(".dt_bias"):
            return self._undo_v_head_reorder(weight, 0, 1)
        return weight
