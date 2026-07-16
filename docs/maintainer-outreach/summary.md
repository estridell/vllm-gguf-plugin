# Maintainer outreach summary

> Draft only. Do not post without maintainer review.

Prism and current upstream ggml both use type ID 42 for Q2_0, but Prism stores
128 values in 34 bytes while upstream stores 64 values in 18 bytes. The codebook
is the same; the scale boundary is not, so a reader using the wrong size table
can silently corrupt weights or read beyond the logical tensor. Type ID,
dimensions, offsets, tags, and aligned file length cannot distinguish them.

Proposed path: PrismML re-exports supported Bonsai models to upstream Q2_0 by
splitting each 128-value block into two 64-value blocks and copying the original
scale, which preserves dequantized values. Keep the pinned Prism compatibility
path only for known legacy artifacts during migration. If Prism needs the
128-value layout for new models, PrismML and ggml should assign it a distinct
maintainer-approved type; no downstream project should choose an ID. A
versioned metadata discriminator is only a possible legacy fallback and needs
explicit agreement.

Reproducer and equal-length GGUF fixtures:
`scripts/gguf_layout_collision.py` and
`tests/fixtures/gguf-layout-collision/`.
