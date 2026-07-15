import gguf


def _check_prism_ternary_layouts() -> None:
    """Fail before gguf-py can interpret tensors with the wrong size table.

    This must remain an import-time check while Prism and upstream assign
    incompatible Q2_0 layouts to type ID 42. Issue #1 tracks a coordinated
    discriminator that would let the check move to per-file loading safely.
    """
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
