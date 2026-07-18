# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType, dequantize

import vllm_gguf_plugin.ops as ops
from vllm_gguf_plugin.quantization import linear

QUANT_TYPE = GGMLQuantizationType.IQ4_NL
# block_iq4_nl: one fp16 scale followed by packed 4-bit indices.
BLOCK_VALUES, BLOCK_BYTES = gguf.GGML_QUANT_SIZES[QUANT_TYPE]
# Batch size above the mmvq cutoff so _fused_mul_mat_gguf takes the
# dequantize-plus-matmul fallback path.
BATCH_SIZE = 32
WORKSPACE_BYTES = 1024 * 1024


@pytest.fixture(autouse=True)
def _seed():
    # tests.utils.seed_everything lives behind an HF snapshot download at
    # import time; seed torch directly instead.
    torch.manual_seed(0)


def make_iq4_nl_qweight(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    """Build a valid random IQ4_NL quantized weight of shape (rows, cols)."""
    assert cols % BLOCK_VALUES == 0
    blocks_per_row = cols // BLOCK_VALUES
    rng = np.random.default_rng(seed)
    scales = rng.normal(0, 0.05, size=(rows, blocks_per_row, 1)).astype(np.float16)
    indices = rng.integers(
        0, 256, size=(rows, blocks_per_row, BLOCK_BYTES - 2), dtype=np.uint8
    )
    blocks = np.concatenate([scales.view(np.uint8), indices], axis=-1)
    return torch.tensor(blocks.reshape(rows, -1), device="cuda")


def assert_matches_reference(
    output: torch.Tensor, x: torch.Tensor, qweight: torch.Tensor
) -> None:
    weight = torch.tensor(
        dequantize(qweight.cpu().numpy(), QUANT_TYPE), device="cuda"
    ).to(x.dtype)
    torch.testing.assert_close(output, x @ weight.T, atol=1e-2, rtol=4e-2)


def bound_workspace(monkeypatch, cols: int) -> int:
    """Shrink the workspace bound to WORKSPACE_BYTES; return max chunk rows."""
    # raising=False so that on an implementation without the bound the tests
    # fail on the oversized dequantize call itself, not on this setattr.
    monkeypatch.setattr(
        linear, "_DEQUANT_MAX_WORKSPACE_BYTES", WORKSPACE_BYTES, raising=False
    )
    return WORKSPACE_BYTES // (cols * torch.float16.itemsize)


@pytest.fixture
def dequantize_spy(monkeypatch):
    """Record the row count of every ops.ggml_dequantize call."""
    row_counts: list[int] = []
    wrapped = ops.ggml_dequantize

    def spy(W, quant_type, m, n, dtype):
        row_counts.append(int(m))
        return wrapped(W, quant_type, m, n, dtype)

    monkeypatch.setattr(ops, "ggml_dequantize", spy)
    return row_counts


@torch.inference_mode()
def test_dequant_workspace_bounded(dequantize_spy, monkeypatch):
    """Tall matrices must be dequantized in row chunks below the workspace
    bound, and the chunked result must match the full dequantization."""
    rows, cols = 4096, 512
    max_rows = bound_workspace(monkeypatch, cols)

    qweight = make_iq4_nl_qweight(rows, cols)
    x = torch.randn(BATCH_SIZE, cols, device="cuda", dtype=torch.float16)

    output = linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert len(dequantize_spy) > 1, "expected multiple dequantization chunks"
    assert sum(dequantize_spy) == rows
    assert all(m <= max_rows for m in dequantize_spy), (
        f"dequantization call exceeded {max_rows} rows: {dequantize_spy}"
    )
    assert output.shape == (BATCH_SIZE, rows)
    assert_matches_reference(output, x, qweight)


@pytest.mark.parametrize("rows_delta, expected_calls", [(-32, 1), (0, 1), (32, 2)])
@torch.inference_mode()
def test_dequant_chunk_boundary(
    dequantize_spy, monkeypatch, rows_delta, expected_calls
):
    """Row counts below and at the chunk size stay single-call; above it
    splits into exactly the needed number of chunks."""
    cols = 512
    max_rows = bound_workspace(monkeypatch, cols)
    rows = max_rows + rows_delta

    qweight = make_iq4_nl_qweight(rows, cols)
    x = torch.randn(BATCH_SIZE, cols, device="cuda", dtype=torch.float16)

    output = linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert len(dequantize_spy) == expected_calls
    assert sum(dequantize_spy) == rows
    assert_matches_reference(output, x, qweight)


@torch.inference_mode()
def test_dequant_small_matrix_single_call(dequantize_spy):
    """With the default workspace bound, ordinary layer shapes keep the
    original single full dequantization call."""
    rows, cols = 2048, 512
    qweight = make_iq4_nl_qweight(rows, cols)
    x = torch.randn(BATCH_SIZE, cols, device="cuda", dtype=torch.float16)

    output = linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    assert dequantize_spy == [rows]
    assert_matches_reference(output, x, qweight)


@torch.inference_mode()
def test_dequant_chunked_matches_unchunked(monkeypatch):
    """Chunked and unchunked paths must agree closely on the same input."""
    rows, cols = 4096, 512
    qweight = make_iq4_nl_qweight(rows, cols)
    x = torch.randn(BATCH_SIZE, cols, device="cuda", dtype=torch.float16)

    unchunked = linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)
    bound_workspace(monkeypatch, cols)
    chunked = linear._fused_mul_mat_gguf(x, qweight, QUANT_TYPE)

    torch.testing.assert_close(chunked, unchunked, atol=1e-3, rtol=1e-3)
