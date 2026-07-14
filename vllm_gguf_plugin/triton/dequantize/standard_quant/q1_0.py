import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_Q1_0, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def q1_0_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 128) * 18
    pos = offs % 128
    packed = tl.load(block_ptrs + 2 + pos // 8, mask=mask, other=0)
    bit = (packed >> (pos % 8)) & 1
    d = load_f16_from_u8(block_ptrs, mask).to(tl.float32)
    out = tl.where(bit == 1, d, -d)
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q1_0_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q1_0_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q1_0)
