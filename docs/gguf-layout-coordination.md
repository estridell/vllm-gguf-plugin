# Q1_0/Q2_0 layout coordination package

This package documents a format collision that cannot be resolved inside a
decoder. PrismML and current upstream ggml assign type ID 41 to compatible
`Q1_0` layouts, but they assign type ID 42 to incompatible `Q2_0` block sizes.
An ID-42 tensor does not contain enough information to select one layout.

The findings below were independently verified on 2026-07-16 against:

- this repository at `8f179856c037d7c4aa246abb6a2fac596b957410`;
- the pinned PrismML llama.cpp and `gguf-py` commit
  [`62061f9`](https://github.com/PrismML-Eng/llama.cpp/commit/62061f91088281e65071cc38c5f69ee95c39f14e);
- current upstream llama.cpp and `gguf-py` commit
  [`505b1ed`](https://github.com/ggml-org/llama.cpp/commit/505b1ed15ca80e2a19f12ff4ac365e40fb374053).

This document separates source-backed facts from proposals. It does not assign
or reserve a GGML type ID, reinterpret upstream ID 42, or define a new metadata
standard.

## Verified physical layouts

The current upstream and pinned Prism enums both define `Q1_0 = 41` and
`Q2_0 = 42`: [upstream enum](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/ggml/include/ggml.h#L431-L432),
[Prism enum](https://github.com/PrismML-Eng/llama.cpp/blob/62061f91088281e65071cc38c5f69ee95c39f14e/ggml/include/ggml.h#L431-L432).
The `gguf-py` size tables disagree only on ID 42:
[upstream sizes](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/gguf-py/gguf/constants.py#L4750-L4751),
[Prism sizes](https://github.com/PrismML-Eng/llama.cpp/blob/62061f91088281e65071cc38c5f69ee95c39f14e/gguf-py/gguf/constants.py#L4588-L4589).

| Definition | Type ID | Values per block | Bytes per block | Scale bytes | Packed bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Upstream Q1_0 | 41 | 128 | 18 | 2 | 16 |
| Prism Q1_0 | 41 | 128 | 18 | 2 | 16 |
| Upstream Q2_0 | 42 | 64 | 18 | 2 | 16 |
| Prism Q2_0 | 42 | 128 | 34 | 2 | 32 |

All byte offsets below are relative to the start of one block. `d` denotes the
stored IEEE 754 binary16 scale converted to the working floating-point type.

### Q1_0, upstream and Prism

Both implementations define a 128-value block as `ggml_half d` followed by
`uint8_t qs[16]`, for 18 physical bytes:
[upstream struct](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/ggml/src/ggml-common.h#L180-L185),
[Prism struct](https://github.com/PrismML-Eng/llama.cpp/blob/62061f91088281e65071cc38c5f69ee95c39f14e/ggml/src/ggml-common.h#L180-L185).

- Bytes 0-1 hold `d`, the binary16 rounding of the float32 mean absolute value
  `sum(abs(x[0:128])) / 128` produced by the reference quantizer.
- Bytes 2-17 hold one sign bit per logical value. Value `i` uses bit
  `i % 8` of byte `2 + floor(i / 8)`, so values are packed least-significant
  bit first in groups of eight.
- A zero bit dequantizes to `-d`; a one bit dequantizes to `+d`. The reference
  quantizer emits one for an input greater than or equal to zero.

The reference quantizer and dequantizer implement these rules directly:
[upstream quantizer](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/ggml/src/ggml-quants.c#L40-L72),
[upstream dequantizer](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/ggml/src/ggml-quants.c#L419-L437).
The corresponding formula is:

```text
q[i] = bit(i) == 1 ? d : -d
```

ID 41 is therefore byte-compatible between the two inspected implementations.

### Upstream Q2_0

Current upstream defines a 64-value block as `ggml_half d` followed by
`uint8_t qs[16]`, for 18 physical bytes:
[upstream struct](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/ggml/src/ggml-common.h#L187-L192).

- Bytes 0-1 hold `d`, the binary16 rounding of the float32 maximum absolute
  value over the 64-value block.
- Bytes 2-17 hold four consecutive 2-bit codes per byte. Value `i` uses bits
  `2 * (i % 4)` and `2 * (i % 4) + 1` of byte
  `2 + floor(i / 4)`, with the first code in the least-significant pair.
- The stored code is `clamp(round(x[i] / d) + 1, 0, 3)`, with a zero reciprocal
  when `d == 0`.
- Code values 0, 1, 2, and 3 represent `-d`, `0`, `+d`, and `+2d`.

The current reference implementation is visible in the
[quantizer](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/ggml/src/ggml-quants.c#L74-L110)
and [dequantizer](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/ggml/src/ggml-quants.c#L439-L457).
The formula is:

```text
q[i] = (code(i) - 1) * d
```

### Prism Q2_0

Pinned Prism defines a 128-value block as `ggml_half d` followed by
`uint8_t qs[32]`, for 34 physical bytes:
[Prism struct](https://github.com/PrismML-Eng/llama.cpp/blob/62061f91088281e65071cc38c5f69ee95c39f14e/ggml/src/ggml-common.h#L187-L192).

The scale representation, 2-bit codebook, least-significant-pair packing, and
dequantization formula match upstream. The physical incompatibility is the
scale boundary: Prism applies one scale to 128 values, while upstream applies
two independent scales to the same 128 values. Pinned Prism `gguf-py` confirms
the `[d, qs]` order and 34-byte block in its
[Q2_0 implementation](https://github.com/PrismML-Eng/llama.cpp/blob/62061f91088281e65071cc38c5f69ee95c39f14e/gguf-py/gguf/quants.py#L657-L693).

## Byte-level collision

The reproducer uses 128 logical values with repeating codes `0, 1, 2, 3`.
Those four codes pack into byte `e4` because the pairs are stored in bit
positions 0, 2, 4, and 6. Binary16 scales `1.0` and `2.0` encode as little-endian
bytes `00 3c` and `00 40`.

```text
Prism ID 42, 128 values, 34 logical bytes:
00 3c | e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4
      | e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4

Upstream ID 42, 128 values, 36 logical bytes:
00 3c | e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4
00 40 | e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4
```

Both generated GGUF files have the same version, tensor count, metadata count,
tensor name, one-dimensional shape `[128]`, type ID 42, relative tensor offset
zero, data offset 96, and total file length 160. Normal alignment and trailing
padding make the physical file length equal. Only the data bytes differ, so
neither the tensor directory nor file length identifies the layout.

Decoding the upstream bytes as Prism uses the first scale for all 128 values
and consumes only 34 of the 36 logical bytes. Values 64-71 become eight values
near `-1` instead of `[-2, 0, 2, 4]` repeated. Decoding the Prism bytes as
upstream treats bytes `e4 e4` at the supposed second scale boundary as
binary16 `-1252`; values 64-71 become
`[1252, 0, -1252, -2504]` repeated, and a reader sized for 36 bytes reads two
bytes beyond the 34-byte logical payload.

The files and exact output are produced by
[`scripts/gguf_layout_collision.py`](../scripts/gguf_layout_collision.py).
The committed files under
[`tests/fixtures/gguf-layout-collision`](../tests/fixtures/gguf-layout-collision)
are deterministic outputs of that script.

## Consequences of choosing the wrong layout

1. **The block count is wrong.** A 128-element tensor has two upstream Q2_0
   blocks but one Prism block, so the reader selects the wrong number of scale
   boundaries before any kernel runs.

2. **The byte length and tensor stride are wrong.** Upstream computes 36 bytes
   per 128 values and Prism computes 34. For rows or adjacent tensors, that
   changes byte strides, expected tensor length, and which bytes belong to each
   block.

3. **Weight corruption can be silent.** Both layouts use valid binary16 fields
   and valid 2-bit codes, so the wrong decoder can produce ordinary floating
   values without an exception. Padding can satisfy an oversized read, and an
   undersized read can simply ignore valid bytes.

4. **Reader behavior is unsafe without prior discrimination.** Current
   `gguf-py` maps type ID to one global `(block_size, type_size)` pair, computes
   `n_bytes = n_elements * type_size // block_size`, and then maps that many
   bytes. See the upstream
   [reader sizing path](https://github.com/ggml-org/llama.cpp/blob/505b1ed15ca80e2a19f12ff4ac365e40fb374053/gguf-py/gguf/gguf_reader.py#L318-L367).
   Since the tensor directory does not carry a per-tensor block size, decoding
   before an external format decision risks over-reading into padding or the
   next region, under-reading the tensor, or silently corrupting values.

Descriptive metadata such as a model name, filename, `general.tags`, or
`general.file_type` is not a versioned physical-layout contract. It may be
useful evidence about provenance, but it cannot make arbitrary ID-42 files safe
to decode.

## Compatibility options

The following are proposals, not settled format decisions.

### Re-export Prism models to current upstream Q2_0

Each Prism 128-value block can be converted to two upstream 64-value blocks by
copying the original binary16 scale into both blocks and splitting the 32 code
bytes into two 16-byte halves. This preserves every dequantized value exactly
and grows each 128 values from 34 to 36 bytes. Recomputing a scale for either
half may improve error but changes values, so that is requantization rather
than a layout-only conversion.

This is the smallest interoperability change because new files would use ID 42
with its current upstream meaning. Existing Prism files remain ambiguous and
need explicit provenance during conversion.

### Assign a new non-conflicting type and name

If the 128-value Prism layout must remain a first-class format, a distinct
maintainer-approved type name and ID would make each tensor self-describing.
Old readers would reject the unknown type instead of silently applying the
upstream Q2_0 layout. This repository must not propose, reserve, or consume a
numeric ID before ggml maintainers accept an assignment.

### Require a versioned metadata discriminator

A mandatory, namespaced, versioned metadata key could identify legacy Prism
files before tensors are sized or mapped. The key needs coordinated ownership,
defined values, file-level or tensor-level scope, and required rejection for a
missing or unknown value. Existing unmarked ID-42 files would still be
ambiguous, and this approach retains two physical meanings for ID 42, so it
requires explicit approval from ggml maintainers rather than a local reader
convention.

### Retain a temporary pinned compatibility implementation

The existing experimental branch pins Prism `gguf-py` to immutable commit
`62061f91088281e65071cc38c5f69ee95c39f14e`. Keeping that environment for
known Bonsai artifacts preserves current loading while coordination proceeds,
but the pin proves which decoder is installed, not which layout an arbitrary
file contains. It must stay temporary and scoped to artifacts whose provenance
is known outside the type ID.

## Recommended paths

**Short term:** PrismML should publish or endorse a layout-only converter and
re-export supported Bonsai models to upstream's 64-value Q2_0 layout. Until
those artifacts exist, this plugin should retain its immutable Prism pin and
current Bonsai behavior only for the known experimental path. No generic
reader should infer Prism layout from ID 42, filenames, or descriptive tags.

**Long term:** Use one physical meaning for upstream `Q2_0` and ID 42. If
PrismML needs to preserve the 128-value format for new files, PrismML and ggml
maintainers should agree on a distinct public type name and ID through the
normal ggml process. A versioned discriminator is a fallback for legacy files,
not a substitute for an unambiguous type in newly exported models.

## Decisions requested from maintainers

### PrismML

1. Confirm that released Prism/Bonsai ID-41 and ID-42 tensors use the layouts
   documented above, and identify the writer commits or releases that emitted
   them.
2. Inventory which published artifacts contain the 128-value ID-42 layout and
   whether every affected file can be identified from trusted provenance.
3. Decide whether supported artifacts will be re-exported to current upstream
   Q2_0, and provide the conversion procedure and migration timeline.
4. If the 128-value format must continue, decide whether PrismML will request a
   new public type name and ID or pursue a coordinated legacy metadata scheme.
5. State how long the pinned Prism implementation and old artifacts need to be
   supported.

### ggml/llama.cpp

1. Confirm that current IDs 41 and 42 and their physical layouts are the
   authoritative upstream definitions intended for GGUF interoperability.
2. Decide whether the Prism 128-value layout is eligible for a distinct public
   type name and ID; any number must come from this maintainer decision.
3. Decide whether a versioned legacy discriminator for existing Prism ID-42
   files is acceptable at all. If it is, define the namespace, values, scope,
   and required behavior when the field is absent or unknown.
4. Clarify whether upstream tools should offer a migration utility or whether
   PrismML should own re-export and conversion.
5. Document the expected treatment of already-published ambiguous files so
   downstream readers do not create incompatible local heuristics.

## Unresolved questions

- The number and names of published Prism/Bonsai files containing the
  128-value ID-42 layout have not been independently inventoried.
- No maintainer-approved type name, numeric ID, or metadata discriminator exists
  for the Prism layout at the time of this verification.
- It is not settled whether ggml maintainers would accept the 128-value layout
  as a separate public type or treat it only as a legacy external format.
- The owner, namespace, scope, and version semantics of any legacy metadata key
  remain undecided.
