# Ternary backend support

This matrix covers the Prism group-128 Q1_0/Q2_0 path. “Compile-only” means
the extension was built for that target but no kernel ran there. “Expected
fallback” means dispatch deliberately avoids the compiled ternary kernel and
uses Triton dequantization followed by framework matmul. It is not a runtime
correctness claim for the target device.

## Backend matrix

The compact cell labels are: **pass** = tested and passing, **compile** =
compile-only, **fallback** = expected fallback, **gated** = unsupported but
cleanly gated, and **unknown** = no evidence.

| Backend | fp16 b1 | fp16 b2 | fp16 b3 | fp16 b4 | fp16 b8 | bf16 b1 | bf16 b2 | bf16 b3 | bf16 b4 | bf16 b8 | Above b8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NVIDIA sm_75 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass at b9; fallback GEMM |
| NVIDIA sm_120 | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass | pass at b9; fallback GEMM |
| CUDA sm_70 | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | unknown | fallback; runtime unknown |
| CUDA sm_80/86/87/89/90/100/110 | compile | compile | compile | compile | compile | compile | compile | compile | compile | compile | fallback; runtime unknown |
| ROCm | fallback | fallback | fallback | fallback | fallback | fallback | fallback | fallback | fallback | fallback | fallback; runtime unknown |
| Triton-only CUDA | pass | fallback | fallback | fallback | pass | pass | fallback | fallback | fallback | pass | pass at b9 via dequantize plus GEMM |
| CPU-only import/reference | gated | gated | gated | gated | gated | gated | gated | gated | gated | gated | gated |

The CUDA architecture list follows `scripts/build_release_wheel.sh`: CUDA 13
builds sm_75/80/86/87/89/90/100/110/120, CUDA 12.8/12.9 builds through
sm_100, and older supported toolkits also include sm_70. The CUDA 13 fatbin
compiled all nine listed targets. sm_70 remains unknown because CUDA 13 no
longer accepts that target and no older toolkit build was run. A successful
fatbin build is only compile evidence; it does not establish numerical
correctness.

## Dispatch and assumptions

- Batches 1 through 8 select compiled MMVQ when the extension is available on
  CUDA. If it is absent, the same calls dequantize with Triton and use framework
  matmul. Batch 9 and above always use dequantize-plus-matmul because ternary
  types have no MMQ implementation in this plugin.
- ROCm deliberately does not select the compiled ternary Q1_0/Q2_0 kernels.
  The copied code has dynamic 32/64-lane warp handling, but this branch has no
  ROCm compile or runtime result for the ternary additions.
- The compiled kernels dispatch float32, fp16, and bf16 activations. Ternary
  block scales and Q8_1 activation scales are fp16, accumulation is fp32, and
  results are converted to the activation dtype.
- CUDA MMVQ uses one hardware warp or wave per output row. Q1_0 and Q2_0 have
  128-value blocks split into four 32-value dot-product chunks. Their 18-byte
  and 34-byte block sizes are guarded by static assertions; both sizes and the
  two-byte scale prefix preserve the two-byte alignment used by packed loads.
- The dot product uses `dp4a`. The release architecture floor is sm_70, above
  CUDA’s sm_61 requirement, so no additional CUDA architecture gate is needed
  for the advertised build list.
- The Triton dequantizers accept fp16, bf16, and fp32 output. They require an
  accelerator tensor and reject CPU tensors with an explicit error; CPU-only
  support is limited to importing the plugin and running the reference codec.
- CUDA compilation uses `-O3 -std=c++17 --use_fast_math -DUSE_CUDA`. ROCm
  compilation omits the nvcc-only fast-math flag, and PyTorch's HIP extension
  path supplies `USE_ROCM`; neither a ROCm compiler nor runtime was available
  for this audit.

## Validation record

CPU-only import and reference/dispatch tests, with the accelerator hidden:

```bash
CUDA_VISIBLE_DEVICES='' python -c \
  'import torch, vllm_gguf_plugin; from vllm_gguf_plugin import ops; print(torch.cuda.is_available(), ops._CUDA_AVAILABLE)'
CUDA_VISIBLE_DEVICES='' pytest -q tests/test_ternary.py -m 'not cuda and not integration'
```

Result: import reported `False False`; 28 tests passed. These tests cover both
activation dtypes at the dispatch boundary, the b1/2/3/4/8 MMVQ cutoff, b9+
fallback selection, ROCm gating, reference quantization, and missing-extension
fallback behavior.

CUDA 13.2 fatbin compile with GCC 15.3:

```bash
CUDA_VISIBLE_DEVICES='' \
TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6;8.7;8.9;9.0;10.0;11.0;12.0' \
python setup.py build_ext --inplace
```

Result: passed for sm_75/80/86/87/89/90/100/110/120. The local CUDA 13.2
toolchain and GCC 15.3 were used because GCC 16 is newer than nvcc supports.

Runtime suite on the local RTX 2070 (sm_75):

```bash
pytest -q tests/test_ternary.py -m cuda
```

Result: 56 tests passed in 6.14 seconds.

Runtime suite on `blackwell-box`, RTX 5060 Ti (sm_120), using the same source
and fatbin mounted read-only in `vllm/vllm-openai:nightly`:

```bash
pytest -q tests/test_ternary.py -m cuda
```

Result: 56 tests passed in 15.28 seconds. Production was restored afterward:
the `gemma` container was healthy, restart count was zero, and
`http://127.0.0.1:8000/v1/models` returned 200.

Pre-commit result: all configured ruff, typos, clang-format, markdownlint, and
filename checks passed.

## Attribution

The vendored CUDA scaffolding derives from llama.cpp commit `b2899`. The Q1_0
and Q2_0 layouts, dequantizers, and CUDA dot-product path are adapted from the
MIT-licensed [`PrismML-Eng/llama.cpp`](https://github.com/PrismML-Eng/llama.cpp/tree/62061f91088281e65071cc38c5f69ee95c39f14e)
fork. The affected headers contain pinned source links, and the redistributed
MIT notice is in `THIRD_PARTY_NOTICES.md`.

The PyTorch reference codec and Triton kernels were implemented in this
project against the Prism group-128 format rather than copied from PrismML’s
CUDA implementation.
