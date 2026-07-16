# Bonsai limitations and format safety

This branch is an experimental, Prism-specific serving path. It is usable for
the named Bonsai group-128 files, but issue
[#1](https://github.com/estridell/vllm-gguf-plugin/issues/1) blocks a safe
general release.

## GGUF type-ID 42 conflict

Upstream ggml and PrismML both assign numeric type ID 42 to `Q2_0`, but their
physical layouts differ:

| Layout | Values per block | Bytes per block | Scale groups |
| --- | ---: | ---: | --- |
| Ordinary upstream Q2_0 | 64 | 18 | Two independent scales per 128 values |
| Prism Q2_0 | 128 | 34 | One scale per 128 values |

The tensor directory records the numeric type, dimensions, and offset, but it
does not identify which block layout produced the bytes. Filenames and
descriptive tags are not a versioned discriminator. Treating the formats as
identical can move scale bytes into packed codes, or packed codes into a
scale, and silently produce incorrect weights.

This repository therefore pins PrismML's `gguf-py` and checks for the Prism
layout at import time. That makes the experiment reproducible, but it does not
resolve the on-disk ambiguity: ordinary upstream Q2_0 GGUF files must not be
loaded in this environment, and Bonsai Q2_0 files must not be decoded by an
ordinary upstream Q2_0 reader. A coordinated non-conflicting type assignment,
mandatory layout discriminator, or explicit conversion path is still needed.
The [layout coordination package](../gguf-layout-coordination.md) records the
byte-level evidence, compatibility options, and unresolved maintainer decisions.

## Current limits

- Bonsai 1.7B, 4B, and 27B group-128 Q2_0 files are the supported artifacts.
  The published group-64 and PQ2_0 variants are not covered by these launch
  instructions.
- Native ternary MMVQ is validated through batch 8. Larger batches use chunked
  dequantization followed by matrix multiplication because ternary MMQ has not
  been ported.
- Q1_0 has exact codec and kernel tests, but the published Bonsai files used in
  validation contain Q2_0 tensors; Q1_0 has not had the same real-model run.
- Only NVIDIA CUDA on `sm_75` and `sm_120` has been tested. ROCm, other CUDA
  architectures, CPU execution, multi-GPU tensor parallelism, and non-Linux
  hosts remain unverified.
- The 27B benchmark covers text-only serving at an 8,192-token context with
  BF16 KV. The advertised full context, multimodal projector, DSpark drafter,
  speculative decoding, and KV-cache quantization remain unverified here.
- The 16 GB 27B launch is close to the GPU limit and intentionally caps active
  sequences at eight. CPU offload is not a safe fallback on a host that also
  has only 16 GB of system RAM.

These limitations describe validation coverage, not a change to the current
decoder. The branch deliberately preserves the pinned Prism dependency,
group-128 interpretation, and fallback behavior while issue #1 remains open.
