# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from transformers import PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from vllm.transformers_utils.config import HFConfigParser
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from .gguf_utils import (
    check_gguf_file,
    is_gguf,
    is_remote_gguf,
    maybe_patch_hf_config_from_gguf,
    split_remote_gguf,
)

QWEN35_GGUF_ARCHITECTURE = "Qwen3_5GGUFForCausalLM"


class GGUFConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        original_model = model
        resolved_model = self._resolve_config_source(model)
        config_dict, config = HFConfigParser().parse(
            resolved_model,
            trust_remote_code=trust_remote_code,
            revision=revision,
            code_revision=code_revision,
            **kwargs,
        )

        if config.model_type == "qwen3_moe" and "norm_topk_prob" not in config_dict:
            config_dict["norm_topk_prob"] = True
            config.update({"norm_topk_prob": True})

        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")

        architecture = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        # Keep the canonical architecture untouched; this parser only runs for
        # GGUF and can therefore select the plugin shim without global effects.
        if architecture == "Qwen3_5ForCausalLM":
            architecture = QWEN35_GGUF_ARCHITECTURE
        config_dict["architectures"] = [architecture]
        config.update({"architectures": [architecture]})

        if is_gguf(original_model):
            config = maybe_patch_hf_config_from_gguf(str(original_model), config)

        return config_dict, config

    @staticmethod
    def _resolve_config_source(model: str | Path) -> str | Path:
        if check_gguf_file(model):
            return Path(model).parent
        if is_remote_gguf(model):
            repo_id, _ = split_remote_gguf(model)
            return repo_id
        return model
