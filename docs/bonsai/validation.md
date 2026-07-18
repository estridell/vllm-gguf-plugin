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
commit `024a566` against vLLM `0.23.1rc1.dev424+g3f5a1e173`. Installing the
plugin left vLLM's canonical `Qwen3_5ForCausalLM` registry entry unchanged, and
the targeted registration tests passed (`2 passed`).

The same source then loaded the real 7.17 GB Bonsai-27B Q2_0 artifact on the
RTX 5060 Ti. vLLM resolved `Qwen3_5GGUFForCausalLM`, loaded 7.06 GiB of model
weights, exposed `bonsai-27b` through `/v1/models`, and completed a deterministic
request (`The capital of Sweden is` → `Stockholm.`). This closes the real-model
validation gap without globally replacing the class used by ordinary non-GGUF
Qwen3.5 and Qwen3.6 models.

See the separate [Bonsai-27B benchmark](benchmarks.md) for the repeated
cross-runtime results and complete methodology.
