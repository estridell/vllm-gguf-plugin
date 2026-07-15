# Install the experimental Bonsai support

The Bonsai path must be installed from this repository's `prism-ternary`
branch in an isolated environment. It depends on vLLM 0.25.1 or newer and
replaces the ordinary `gguf` package with PrismML's GGUF reader, so it should
not share an environment with a general GGUF deployment.

## Prerequisites

- Python 3.10 or newer
- A working NVIDIA CUDA toolkit and driver
- Git and [uv](https://docs.astral.sh/uv/)

Install `uv` if needed, clone the branch, and install the plugin:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone --branch prism-ternary --single-branch \
  https://github.com/estridell/vllm-gguf-plugin.git
cd vllm-gguf-plugin
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e . --torch-backend=auto
```

The plugin's [`pyproject.toml`](../../pyproject.toml) pins the Prism reader to
this exact immutable dependency:

```text
gguf @ git+https://github.com/PrismML-Eng/llama.cpp@62061f91088281e65071cc38c5f69ee95c39f14e#subdirectory=gguf-py
```

Do not replace that commit with PrismML's floating `prism` branch or a normal
upstream `gguf` release. The import-time layout check requires Prism Q1_0 to be
128 values in 18 bytes and Prism Q2_0 to be 128 values in 34 bytes; a different
reader either fails immediately or risks decoding type ID 42 incorrectly.

Verify the installed layout before downloading a model:

```bash
python - <<'PY'
import gguf

for name in ("Q1_0", "Q2_0"):
    qtype = getattr(gguf.GGMLQuantizationType, name)
    print(name, int(qtype), gguf.GGML_QUANT_SIZES[qtype])
PY
```

The expected output is:

```text
Q1_0 41 (128, 18)
Q2_0 42 (128, 34)
```

Continue with the [launch examples](usage.md). The model references there name
the exact group-128 files and deliberately avoid the similarly named group-64
artifacts.
