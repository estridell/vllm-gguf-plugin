# Draft ggml/llama.cpp design issue: Q2_0 ID collision coordination

> Draft only. Do not post without PrismML context and maintainer review.

## Title

Design decision: migration path for legacy Prism Q2_0 files using type ID 42

## Body

We are coordinating an existing GGUF type collision before upstreaming Prism
ternary support in `vllm-gguf-plugin`. This is a format-design question, not a
request to reinterpret upstream ID 42 or reserve a number downstream.

At current upstream commit
`505b1ed15ca80e2a19f12ff4ac365e40fb374053`, IDs 41 and 42 are:

- Q1_0 ID 41: 128 values in 18 bytes, with a binary16 mean-absolute scale and
  16 sign bytes.
- Q2_0 ID 42: 64 values in 18 bytes, with a binary16 absolute-maximum scale and
  16 bytes containing four 2-bit codes each.

Pinned Prism commit `62061f91088281e65071cc38c5f69ee95c39f14e`
uses the same ID 41 layout, but its ID 42 Q2_0 block contains 128 values in 34
bytes: one binary16 scale followed by 32 code bytes. The Q2_0 packing and
formula `(code - 1) * d` match upstream, while the scale group and block size do
not.

A GGUF tensor entry records dimensions, type ID, and an offset, but no physical
block size. Our minimal fixtures have identical tensor directories and equal
aligned file lengths. Reading Prism bytes with the upstream size table treats
quant bytes as the second binary16 scale and reads beyond the logical tensor;
reading upstream bytes with the Prism size table consumes the second scale as
codes and ignores valid bytes. Both paths can return floats without an error.

We would like maintainer decisions on these points:

1. Are current IDs 41 and 42 and their inspected layouts the authoritative
   definitions intended for GGUF interoperability?
2. If PrismML needs to keep its 128-value format, should it request a distinct
   public type name and ID through the normal ggml allocation process?
3. Would a mandatory namespaced and versioned discriminator be acceptable for
   already-published legacy Prism ID-42 files, or should those files only be
   converted out of band? If metadata is acceptable, what namespace, scope,
   values, and missing-field behavior should apply?
4. Should an upstream migration utility be considered, or should PrismML own
   layout-only conversion and model re-export?
5. What guidance should downstream readers follow for ambiguous legacy ID-42
   files so they do not create incompatible filename or tag heuristics?

Our recommendation is to keep one physical meaning for upstream Q2_0 and ID 42.
In the short term, PrismML can split each 128-value legacy block into two
64-value blocks, copy the original binary16 scale to both, and split the code
bytes; this preserves dequantized values exactly. If the 128-value format needs
to continue for new files, it should use a separately approved type rather than
metadata overloading ID 42. Metadata would remain a legacy migration mechanism.

Reproducer and generated fixtures are ready to attach:

```text
python scripts/gguf_layout_collision.py --output-dir /tmp/gguf-layout-fixtures
```

The full source-backed comparison is in
`docs/gguf-layout-coordination.md`. No numeric ID is proposed by the downstream
project.
