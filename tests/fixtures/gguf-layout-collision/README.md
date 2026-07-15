# GGUF layout collision fixtures

These three 160-byte GGUF v3 files are deterministic outputs of:

```bash
python scripts/gguf_layout_collision.py \
  --output-dir tests/fixtures/gguf-layout-collision
```

`q1_0-id41.gguf` contains the shared upstream/Prism Q1_0 layout.
`q2_0-id42-upstream.gguf` contains two upstream 64-value blocks, while
`q2_0-id42-prism.gguf` contains one Prism 128-value block. The two ID-42 files
have identical tensor directories, data offsets, and total lengths; only their
data bytes differ.

The script prints the SHA-256 digest of every generated file and demonstrates
both wrong Q2_0 interpretations. The unit test regenerates the structures in
memory and checks the exact bytes and misdecoded values.
