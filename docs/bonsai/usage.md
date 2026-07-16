# Launch Bonsai models

These commands use the exact group-128 Q2_0 filenames published by PrismML.
The separate `--tokenizer` repositories also provide the Hugging Face config
that vLLM needs; omitting them makes the GGUF-only repository insufficient for
model construction.

## Bonsai 1.7B

```bash
vllm serve \
  prism-ml/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf \
  --tokenizer prism-ml/Ternary-Bonsai-1.7B-unpacked \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --seed 0
```

## Bonsai 4B

```bash
vllm serve \
  prism-ml/Ternary-Bonsai-4B-gguf/Ternary-Bonsai-4B-Q2_0.gguf \
  --tokenizer prism-ml/Ternary-Bonsai-4B-unpacked \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --generation-config vllm \
  --seed 0
```

## Bonsai 27B on a 16 GB GPU

```bash
vllm serve \
  prism-ml/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf \
  --served-model-name bonsai-27b \
  --tokenizer prism-ml/Ternary-Bonsai-27B-unpacked \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --kv-cache-dtype auto \
  --mamba-ssm-cache-dtype float32 \
  --generation-config vllm \
  --seed 0
```

The 27B configuration above is the tested 16 GB launch shape: 8,192 tokens,
at most eight active sequences, and float32 Mamba state. On the tested RTX
5060 Ti, `--kv-cache-dtype auto` selected BF16 KV. Keep CUDA graphs enabled by
leaving out `--enforce-eager`; they were required for acceptable 27B decode
performance in testing.

Weights, runtime allocations, and KV cache leave little headroom on a 16 GB
GPU. If startup fails, lower `--max-model-len` or `--max-num-seqs`; do not add
`--cpu-offload-gb` on a 16 GB host, because moving model pages into constrained
system RAM can make the whole host unresponsive. When using Docker on such a
host, also cap the container with `--memory 12g --memory-swap 12g` so a failed
load is killed inside the container.

The native ternary MMVQ path is selected through decode batch 8. Above batch 8
the current decoder preserves correctness by dequantizing weights in chunks
and using the ordinary matrix multiply fallback, but it uses more temporary
memory and is slower. Ternary MMQ tiles have not been ported, so raising
`--max-num-seqs` above 8 is not a performance optimization on this branch.

Only use the filenames shown above. In particular, files ending in `_g64.gguf`
or `Q2_g64.gguf` use a different layout and are outside this branch's Bonsai
support.
