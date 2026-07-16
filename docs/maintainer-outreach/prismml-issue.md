# Draft PrismML issue: resolve legacy Q2_0 GGUF layout collision

> Draft only. Do not post without maintainer review.

## Title

Coordinate migration of Prism Q2_0 files that reuse upstream GGML type ID 42

## Body

We are preparing the Prism ternary work in `vllm-gguf-plugin` for upstream and
found a physical-layout collision that cannot be resolved from a GGUF tensor
entry.

Verified against Prism commit
`62061f91088281e65071cc38c5f69ee95c39f14e` and current upstream llama.cpp
commit `505b1ed15ca80e2a19f12ff4ac365e40fb374053`:

| Definition | ID | Values/block | Bytes/block | Physical layout |
| --- | ---: | ---: | ---: | --- |
| Prism Q1_0 | 41 | 128 | 18 | binary16 scale + 16 sign bytes |
| Upstream Q1_0 | 41 | 128 | 18 | identical to Prism |
| Prism Q2_0 | 42 | 128 | 34 | binary16 scale + 32 2-bit-code bytes |
| Upstream Q2_0 | 42 | 64 | 18 | binary16 scale + 16 2-bit-code bytes |

The Q2_0 codebook and formula match: codes 0-3 dequantize as
`(code - 1) * d`. The scale boundary does not. For 128 values, a Prism reader
expects one 34-byte block while upstream expects two 18-byte blocks. Both files
can have identical tensor directories and equal aligned file lengths, so ID 42,
dimensions, offsets, tags, and file size cannot safely select a decoder.

We have a self-contained reproducer and generated GGUF fixtures showing both
silent corruption directions. No runtime behavior change is proposed in this
coordination issue.

Could PrismML please decide and confirm:

1. Which released Bonsai/Prism artifacts use the 128-value ID-42 layout, and
   which writer commit or release produced them?
2. Can supported models be re-exported to upstream's current 64-value Q2_0
   layout? A layout-only conversion can split each Prism block into two blocks,
   copy the binary16 scale to both, and split the code bytes, preserving every
   dequantized value.
3. If the 128-value layout must continue for new files, will PrismML request a
   distinct public type name and ID through ggml rather than reuse ID 42?
4. Is a namespaced, versioned metadata discriminator needed only for legacy
   files, and can all affected legacy files be identified from trusted
   provenance?
5. What migration and compatibility window should downstreams plan for?

Our proposed short-term path is to re-export supported artifacts to upstream
Q2_0 and keep the current immutable Prism dependency only for known legacy
Bonsai files during migration. The proposed long-term path is one physical
meaning for upstream ID 42, with a separately approved type if Prism needs to
continue the 128-value format.

References and attachments to add before posting:

- coordination document: `docs/gguf-layout-coordination.md`
- reproducer: `scripts/gguf_layout_collision.py`
- fixtures: `tests/fixtures/gguf-layout-collision/`
- upstream Q2_0 introduction: ggml-org/llama.cpp commit `bec4772`
- Prism Q2_0 introduction: PrismML-Eng/llama.cpp commit `984bf97`
