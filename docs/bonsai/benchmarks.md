# Bonsai-27B benchmark

The metric is effective output throughput over the complete client-side wall
clock. vLLM leads at concurrency 1 through 4, while PrismML's llama.cpp fork
leads once all eight server slots are active. These results do not support a
claim that either runtime is universally faster.

| Concurrency | vLLM output tok/s | PrismML llama.cpp output tok/s | Lead |
| ---: | ---: | ---: | --- |
| 1 | 46.28 | 39.41 | vLLM |
| 2 | 52.75 | 18.62 | vLLM |
| 4 | 58.45 | 26.51 | vLLM |
| 8 | 61.48 | 73.65 | PrismML llama.cpp |

## Method

Both runtimes served the identical Bonsai-27B Q2_0 GGUF and received the same
fixed, pre-rendered prompts through `/v1/completions`. Every request used
greedy sampling and produced exactly 256 tokens. Both servers used an
8,192-token context, BF16 KV cache, disabled prompt caching, and eight active
slots.

Each measured run began from a loaded idle server, performed four unmeasured
warm-ups, and then timed the full request batch with a client-side monotonic
wall clock. Each concurrency point was repeated; the reported values are the
medians of those repeated measurements. This method includes queueing, prompt
processing, and generation rather than comparing a decode-only metric from one
runtime with an end-to-end metric from the other.
