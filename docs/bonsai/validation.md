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

## Qwen3.5/3.6 registration isolation

The GGUF-only architecture selector was revalidated on 2026-07-18 from plugin
commit `024a566` against the supported vLLM `0.25.1` image
`vllm/vllm-openai@sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089`.
Installing the plugin left vLLM's canonical `Qwen3_5ForCausalLM` registry entry
unchanged, and the targeted registration tests passed (`2 passed`).

The same source then loaded
`Ternary-Bonsai-27B-Q2_0.gguf` (SHA-256
`868c11714cf8fe47f5ec9eeb2be0ab1a337112886f92ee0ede6b855c4fa31757`)
on the RTX 5060 Ti with an 8,192-token context, eight active sequences, BF16 KV,
90% GPU-memory utilization, and seed 0. vLLM resolved
`Qwen3_5GGUFForCausalLM`, reported 7.06 GiB used for model loading, and exposed
`bonsai-27b` through `/v1/models`. A greedy smoke request with prompt
`The capital of Sweden is`, `temperature=0`, `seed=0`, and `max_tokens=16`
returned HTTP 200 and began `Stockholm.`. This closes the real-model loading and
serving validation gap without globally replacing the class used by ordinary
non-GGUF Qwen3.5 and Qwen3.6 models; it is not a broad model-quality claim.

See the separate [Bonsai-27B benchmark](benchmarks.md) for the repeated
cross-runtime results and complete methodology.
