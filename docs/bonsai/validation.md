# Bonsai validation and benchmark results

The experimental path has been exercised on two NVIDIA generations:

| GPU | CUDA architecture | Scope |
| --- | --- | --- |
| RTX 2070 | `sm_75` | 1.7B and 4B correctness, kernel, and memory work |
| RTX 5060 Ti 16 GB | `sm_120` | 27B load, serving, and concurrency benchmark |

## Correctness evidence

The Q1_0 and Prism Q2_0 reference codecs are covered by generated fixtures.
For Q2_0, Prism `gguf-py` packing and dequantization match the repository's
Torch reference arrays exactly. Triton and CUDA dequantization also match that
reference exactly, and packed model tensor loading was checked against the
Prism path. End-to-end perplexity differed by approximately +0.04% from
PrismML's llama.cpp fork in the validation run.

The hermetic CPU tests require no model download:

```bash
pytest tests/test_ternary.py -m "not cuda and not integration"
```

CUDA kernel coverage is explicit:

```bash
pytest tests/test_ternary.py -m cuda
```

Optional real-model checks accept local model paths:

```bash
pytest tests/integration/test_ternary_models.py \
  --ternary-model /path/to/Ternary-Bonsai-1.7B-Q2_0.gguf \
  --ternary-4b-model /path/to/Ternary-Bonsai-4B-Q2_0.gguf \
  --ternary-4b-config /path/to/Ternary-Bonsai-4B-unpacked
```

See the separate [Bonsai-27B benchmark](benchmarks.md) for the repeated
cross-runtime results and complete methodology.
