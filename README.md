# vLLM GGUF Quantization Plugin

This plugin provides out-of-tree GGUF quantization support for vLLM after
in-tree support deprecation
([vllm-project/vllm#39583](https://github.com/vllm-project/vllm/issues/39583)).

## Experimental Bonsai/Prism ternary support

The `prism-ternary` branch adds the Prism Q1_0 and Q2_0 group-128 tensor
layouts, packed ternary embeddings and output heads, CUDA dequantization and
MMVQ kernels, and the Qwen adapters needed to serve the Bonsai 1.7B, 4B, and
27B GGUF models. This path is experimental: it has been tested on an RTX 2070
(`sm_75`) and RTX 5060 Ti (`sm_120`), but the unresolved GGUF type-ID 42
conflict prevents it from becoming the general plugin dependency.

Start with the [Bonsai installation guide](docs/bonsai/installation.md), then
use the exact [launch examples](docs/bonsai/usage.md). The detailed notes cover
[correctness validation](docs/bonsai/validation.md), the
[Bonsai-27B benchmark](docs/bonsai/benchmarks.md), and
[format safety, resource limits, and unverified areas](docs/bonsai/limitations.md).

## Installation

### Prerequisites

- CUDA toolkit or ROCm toolkit

We recommend [uv](https://docs.astral.sh/uv/) for package management. If you
don't have it installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### From Source

1. Clone this repository:

   ```bash
   git clone https://github.com/vllm-project/vllm-gguf-plugin
   cd vllm-gguf-plugin
   ```

2. Install the plugin in development mode:

   ```bash
   uv pip install -e . --torch-backend=auto
   ```

Or install directly:

```bash
uv pip install . --torch-backend=auto
```

## Development

```bash
uv pip install -e .[dev] --torch-backend=auto
pre-commit install
pre-commit run --all-files
```

The same hooks also run in GitHub Actions on every push and pull request.

The Prism ternary branch currently depends on vLLM 0.25.1 or newer because its
GGUF-only Qwen3.5/3.6 shim subclasses vLLM's shipped `Qwen3_5ForCausalLM` and
uses its hybrid-state hooks. The shim also works around missing GGUF quantizer
propagation to Qwen3.5 token embeddings; that behavior should move into vLLM
core if it is generally correct.

The ternary unit tests use generated fixtures and require no model downloads:

```bash
pytest tests/test_ternary.py -m "not cuda and not integration"
```

CUDA kernel coverage is explicit and serialized:

```bash
pytest tests/test_ternary.py -m cuda
```

The optional real-model tests require local assets and skip when a configured
path is missing. They can be run with pytest options:

```bash
pytest tests/integration/test_ternary_models.py \
  --ternary-model /path/to/ternary-model.gguf \
  --ternary-4b-model /path/to/bonsai-4b.gguf \
  --ternary-4b-config /path/to/bonsai-4b-config
```

The equivalent environment variables are `VLLM_GGUF_TEST_TERNARY_MODEL`,
`VLLM_GGUF_TEST_TERNARY_4B_MODEL`, and
`VLLM_GGUF_TEST_TERNARY_4B_CONFIG`.

The tested and compile-only ternary backend matrix, dispatch rules, and known
gaps are recorded in [docs/ternary-backend-support.md](docs/ternary-backend-support.md).

## Usage

```bash
vllm serve Qwen/Qwen3-0.6B-GGUF:Q8_0 --tokenizer Qwen/Qwen3-0.6B
```
