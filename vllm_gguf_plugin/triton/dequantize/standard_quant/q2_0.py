import torch
import triton
import triton.language as tl

from ...gemm.utils import GGML_TYPE_Q2_0, load_f16_from_u8
from ..utils import dequant_offsets, run_dequantize_kernel


@triton.jit
def q2_0_dequantize_kernel(w_ptr, y_ptr, total, BLOCK_SIZE: tl.constexpr):
    offs, mask = dequant_offsets(total, BLOCK_SIZE)
    block_ptrs = w_ptr + (offs // 128) * 34
    pos = offs % 128
    packed = tl.load(block_ptrs + 2 + pos // 4, mask=mask, other=0)
    code = (packed >> (2 * (pos % 4))) & 3
    d = load_f16_from_u8(block_ptrs, mask).to(tl.float32)
    out = (code.to(tl.float32) - 1.0) * d
    tl.store(y_ptr + offs, out, mask=mask)


def ggml_dequantize_q2_0_triton(
    W: torch.Tensor,
    m: int,
    n: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return run_dequantize_kernel(q2_0_dequantize_kernel, W, m, n, dtype, GGML_TYPE_Q2_0)
