import gguf


def _check_prism_ternary_layouts() -> None:
    expected = {"Q1_0": (128, 18), "Q2_0": (128, 34)}
    actual = {
        name: gguf.GGML_QUANT_SIZES.get(getattr(gguf.GGMLQuantizationType, name, None))
        for name in expected
    }
    if actual != expected:
        raise ImportError(
            "vllm-gguf-plugin requires PrismML gguf-py with group-128 "
            f"ternary layouts {expected}, but found {actual}. Upstream ggml "
            "reuses Q2_0 type id 42 for an incompatible group-64 layout; "
            "install the PrismML llama.cpp prism branch dependency."
        )


_check_prism_ternary_layouts()
