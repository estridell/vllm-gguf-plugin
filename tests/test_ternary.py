import gc
import os
from collections import Counter
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
import vllm.model_executor.layers.vocab_parallel_embedding as vocab_embedding_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from vllm_gguf_plugin import ops
from vllm_gguf_plugin.gguf_utils import is_valid_gguf_quant_type
from vllm_gguf_plugin.quantization.config import GGUFConfig
from vllm_gguf_plugin.quantization.linear import (
    _DEQUANT_ROW_CHUNK_SIZE,
    _fused_mul_mat_gguf,
    GGUFLinearMethod,
)
from vllm_gguf_plugin.quantization.vocal_embeds import _apply_gguf_embedding
from vllm_gguf_plugin.quantization.utils import DEQUANT_TYPES
from vllm_gguf_plugin.reference.ternary import (
    dequant_q1_0,
    dequant_q2_0,
    quantize_q1_0,
    quantize_q2_0,
)
from vllm_gguf_plugin.triton.dequantize.interface import ggml_dequantize_triton
from vllm_gguf_plugin.weight_utils import (
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator,
)

Q1_0 = gguf.GGMLQuantizationType.Q1_0
Q2_0 = gguf.GGMLQuantizationType.Q2_0
BONSAI_MODEL = Path("/home/stridell/bonsai/models/Ternary-Bonsai-1.7B-Q2_0.gguf")
BONSAI_4B_MODEL = Path("/home/stridell/bonsai/models/Ternary-Bonsai-4B-Q2_0.gguf")
BONSAI_4B_CONFIG = Path("/home/stridell/bonsai/models/bonsai-4b")


@pytest.mark.parametrize("shape", [(128,), (3, 256), (2, 3, 128)])
def test_q2_0_matches_prism_numpy(shape: tuple[int, ...]) -> None:
    values = np.random.default_rng(20260714).standard_normal(shape).astype(np.float32)
    prism_packed = gguf.quants.quantize(values, Q2_0)

    torch_packed = quantize_q2_0(torch.from_numpy(values))
    assert np.array_equal(torch_packed.numpy(), prism_packed)

    prism_dequant = gguf.quants.dequantize(prism_packed, Q2_0)
    torch_dequant = dequant_q2_0(torch.from_numpy(prism_packed), shape)
    assert np.array_equal(torch_dequant.numpy(), prism_dequant)


def test_q2_0_dequantizes_valid_code_three() -> None:
    packed = torch.zeros(34, dtype=torch.uint8)
    packed[:2] = torch.tensor([0x00, 0x3C], dtype=torch.uint8)  # fp16 1.0
    packed[2:] = 0x55  # all remaining codes are 1 (zero)
    packed[2] = 0x57  # codes [3, 1, 1, 1]

    output = dequant_q2_0(packed, (128,))
    assert output[0].item() == 2.0
    assert torch.count_nonzero(output[1:]) == 0

    prism_output = gguf.quants.dequantize(packed.numpy(), Q2_0)
    assert np.array_equal(output.numpy(), prism_output)


def test_q1_0_quantize_dequantize_and_hand_computed_block() -> None:
    values = torch.cat((torch.full((64,), -2.0), torch.full((64,), 2.0)))
    packed = quantize_q1_0(values)

    expected = torch.tensor(
        [0x00, 0x40, *([0x00] * 8), *([0xFF] * 8)], dtype=torch.uint8
    )
    assert torch.equal(packed, expected)
    assert torch.equal(dequant_q1_0(packed, values.shape), values)


def test_q1_0_uses_mean_absolute_scale_and_nonnegative_sign_bit() -> None:
    values = torch.arange(-64, 64, dtype=torch.float32)
    packed = quantize_q1_0(values)
    output = dequant_q1_0(packed, values.shape)
    expected_scale = values.abs().sum(dtype=torch.float32) / 128

    assert (
        output.abs().unique().item() == expected_scale.to(torch.float16).float().item()
    )
    assert output[63] < 0
    assert output[64] > 0  # Prism encodes zero as the positive sign.


@pytest.mark.parametrize(
    ("qtype", "quantize", "dequantize"),
    [(Q1_0, quantize_q1_0, dequant_q1_0), (Q2_0, quantize_q2_0, dequant_q2_0)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_triton_dequant_matches_torch_reference(qtype, quantize, dequantize, dtype):
    if not torch.cuda.is_available():
        pytest.skip("Triton ternary kernels require CUDA")
    values = torch.randn((3, 256), generator=torch.Generator().manual_seed(11))
    packed = quantize(values)
    reference = dequantize(packed, values.shape).to(dtype)

    output = ggml_dequantize_triton(
        packed.cuda(), int(qtype), values.shape[0], values.shape[1], dtype
    )
    assert torch.equal(output.cpu(), reference)


@pytest.mark.parametrize(
    ("qtype", "quantize", "dequantize"),
    [(Q1_0, quantize_q1_0, dequant_q1_0), (Q2_0, quantize_q2_0, dequant_q2_0)],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cuda_dequant_matches_torch_reference_exactly(
    qtype, quantize, dequantize, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA ternary dequantization requires CUDA")
    assert ops._cuda_kernel_available("ggml_dequantize", int(qtype))
    values = torch.randn((5, 512), generator=torch.Generator().manual_seed(31))
    packed = quantize(values)
    reference = dequantize(packed, values.shape).to(dtype)

    output = ops.ggml_dequantize(
        packed.cuda(), int(qtype), values.shape[0], values.shape[1], dtype
    )
    assert torch.equal(output.cpu(), reference)


@pytest.mark.parametrize(
    ("qtype", "quantize", "dequantize"),
    [(Q1_0, quantize_q1_0, dequant_q1_0), (Q2_0, quantize_q2_0, dequant_q2_0)],
)
def test_ternary_embedding_gathers_packed_rows_then_dequantizes_exactly(
    qtype, quantize, dequantize
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA ternary embedding lookup requires CUDA")
    values = torch.randn((11, 256), generator=torch.Generator().manual_seed(41))
    packed = quantize(values)
    indices = torch.tensor([[7, 1, 7], [0, 10, 3]], device="cuda")

    output = _apply_gguf_embedding(
        indices,
        packed.cuda(),
        int(qtype),
        values.shape[1],
        torch.float16,
    )
    reference = dequantize(packed, values.shape).index_select(
        0, indices.cpu().flatten()
    )
    reference = reference.view(*indices.shape, values.shape[1]).to(torch.float16)

    assert torch.equal(output.cpu(), reference)


def test_ternary_parallel_lm_head_stays_packed_and_uses_linear_apply(
    monkeypatch,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA ternary lm_head requires CUDA")
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_rank", lambda: 0
    )
    monkeypatch.setattr(
        vocab_embedding_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    values = torch.randn((7, 256), generator=torch.Generator().manual_seed(43))
    packed = quantize_q2_0(values)

    with torch.device("cuda"):
        layer = ParallelLMHead(
            num_embeddings=7,
            embedding_dim=256,
            org_num_embeddings=7,
            padding_size=1,
            params_dtype=torch.float16,
            quant_config=GGUFConfig(),
        )
    layer.qweight.weight_loader(layer.qweight, packed)
    layer.qweight_type.weight_loader(
        layer.qweight_type, torch.tensor(int(Q2_0), dtype=torch.uint8)
    )
    layer.quant_method.process_weights_after_loading(layer)

    assert layer.qweight.dtype == torch.uint8
    assert layer.qweight.shape == packed.shape
    assert type(layer.quant_method).apply is GGUFLinearMethod.apply
    x = torch.randn((1, 256), device="cuda", dtype=torch.float16)
    output = layer.quant_method.apply(layer, x)
    reference = x.float() @ dequant_q2_0(packed, values.shape).cuda().T
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1.0)


def test_bonsai_4b_llm_weight_load_stays_within_packed_budget() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Bonsai 4B memory assertion requires CUDA")
    assert BONSAI_4B_MODEL.is_file(), f"Missing test model: {BONSAI_4B_MODEL}"
    assert BONSAI_4B_CONFIG.is_dir(), f"Missing model config: {BONSAI_4B_CONFIG}"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.pop("VLLM_GGUF_USE_CUDA", None)
    gc.collect()
    torch.cuda.empty_cache()

    from vllm import LLM

    llm = LLM(
        model=str(BONSAI_4B_MODEL),
        tokenizer="Qwen/Qwen3-4B",
        hf_config_path=str(BONSAI_4B_CONFIG),
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        dtype="half",
        seed=0,
    )
    try:
        runner = llm.llm_engine.model_executor.driver_worker.worker.model_runner
        expected_packed_bytes = sum(
            module.qweight.numel()
            for module in runner.model.modules()
            if hasattr(module, "qweight")
            and hasattr(module, "qweight_type")
            and int(module.qweight_type.weight_type) in (int(Q1_0), int(Q2_0))
        )
        assert expected_packed_bytes > 0
        assert runner.model_memory_usage < expected_packed_bytes * 1.15
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.parametrize(
    ("qtype", "quantize", "dequantize"),
    [(Q1_0, quantize_q1_0, dequant_q1_0), (Q2_0, quantize_q2_0, dequant_q2_0)],
)
@pytest.mark.parametrize("rows,cols", [(3, 2048), (5, 5120), (7, 17408)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_ternary_gemv_matches_fp32_reference(
    qtype, quantize, dequantize, rows, cols, dtype
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA ternary GEMV requires CUDA")
    assert ops._cuda_kernel_available("ggml_mul_mat_vec_a8", int(qtype))
    generator = torch.Generator().manual_seed(rows * 100_000 + cols)
    weight = torch.randn((rows, cols), generator=generator)
    packed = quantize(weight)
    x = (0.25 * torch.randn((1, cols), generator=generator)).to(dtype).cuda()

    output = ops.ggml_mul_mat_vec_a8(packed.cuda(), x, int(qtype), rows)
    reference = x.float() @ dequantize(packed, weight.shape).cuda().T
    torch.testing.assert_close(output.float(), reference, rtol=1e-2, atol=1.0)


def test_large_dequant_fallback_chunks_output_rows(monkeypatch) -> None:
    rows = _DEQUANT_ROW_CHUNK_SIZE + 1
    block_size, type_size = gguf.GGML_QUANT_SIZES[Q2_0]
    qweight = torch.zeros((rows, type_size), dtype=torch.uint8)
    x = torch.ones((9, block_size), dtype=torch.float32)
    dequant_rows = []

    def fake_dequantize(weight, quant_type, m, n, dtype):
        assert quant_type == int(Q2_0)
        assert m == weight.shape[0]
        assert n == block_size
        assert dtype == x.dtype
        dequant_rows.append(m)
        return torch.ones((m, n), dtype=dtype)

    monkeypatch.setattr(ops, "ggml_dequantize", fake_dequantize)

    output = _fused_mul_mat_gguf(x, qweight, int(Q2_0))

    assert dequant_rows == [_DEQUANT_ROW_CHUNK_SIZE, 1]
    assert output.shape == (9, rows)
    assert torch.all(output == block_size)


def test_cuda_ternary_gemv_ds_y_and_iqs_chunk_conventions() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA ternary GEMV requires CUDA")

    # All-negative Q1_0 makes the integer dot term zero, so the result depends
    # only on ds.y. The small values deliberately quantize to zero beside the
    # outlier; using the sum of quantized activations would produce -127 rather
    # than the pre-quantization sum near -189.
    q1_weight = -torch.ones((1, 128), dtype=torch.float32)
    q1_packed = quantize_q1_0(q1_weight).cuda()
    q1_x = torch.full((1, 128), 0.49, dtype=torch.float16, device="cuda")
    q1_x[0, 0] = 127.0
    q1_output = ops.ggml_mul_mat_vec_a8(q1_packed, q1_x, int(Q1_0), 1)
    q1_reference = (
        q1_x.float() @ dequant_q1_0(q1_packed, q1_weight.shape).T
    ).to(torch.float16)
    assert torch.equal(q1_output, q1_reference)
    assert q1_output.item() < -180.0

    # One Q2_0 block with a distinct constant code in each consecutive
    # 32-weight chunk proves iqs visits chunks 0,1,2,3 in order. Codes map to
    # symbols -1,0,+1,+2 and activations are 1,2,3,4, so the dot is 320.
    q2_packed = torch.empty((1, 34), dtype=torch.uint8)
    q2_packed[0, :2] = torch.tensor([1.0], dtype=torch.float16).view(torch.uint8)
    q2_packed[0, 2:10] = 0x00
    q2_packed[0, 10:18] = 0x55
    q2_packed[0, 18:26] = 0xAA
    q2_packed[0, 26:34] = 0xFF
    q2_x = torch.cat(
        [torch.full((32,), value, dtype=torch.float16) for value in (1, 2, 3, 4)]
    ).reshape(1, 128).cuda()
    q2_output = ops.ggml_mul_mat_vec_a8(q2_packed.cuda(), q2_x, int(Q2_0), 1)
    assert q2_output.item() == 320.0


def test_dequant_fallback_gemm_uses_group_128_shape_math() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Triton ternary kernels require CUDA")
    values = torch.randn((3, 256), generator=torch.Generator().manual_seed(12))
    packed = quantize_q2_0(values).cuda()
    # The ternary MMVQ threshold is eight rows; batch nine deliberately keeps
    # this coverage on the dequantize-plus-GEMM fallback.
    x = torch.randn((9, 256), device="cuda", dtype=torch.float16)

    output = _fused_mul_mat_gguf(x, packed, int(Q2_0))
    expected = x @ dequant_q2_0(packed, values.shape).to(torch.float16).T
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("batch", [3, 4, 8])
def test_ternary_mmvq_covers_decode_batches(batch: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA ternary GEMV requires CUDA")
    values = torch.randn((512, 640), generator=torch.Generator().manual_seed(21))
    packed = quantize_q2_0(values).cuda()
    x = torch.randn((batch, 640), device="cuda", dtype=torch.float16)

    output = _fused_mul_mat_gguf(x, packed, int(Q2_0))
    expected = x @ dequant_q2_0(packed, values.shape).cuda().to(torch.float16).T
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1.0)


def test_mini_gguf_q2_0_recognized_by_reader_and_weight_iterator(tmp_path) -> None:
    path = tmp_path / "mini-Q2_0.gguf"
    values = np.linspace(-1, 1, 256, dtype=np.float32).reshape(2, 128)
    packed = gguf.quants.quantize(values, Q2_0)
    writer = gguf.GGUFWriter(path, "test")
    writer.add_tensor("test.weight", packed, raw_dtype=Q2_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    tensor = gguf.GGUFReader(path).tensors[0]
    assert tensor.tensor_type == Q2_0
    assert int(tensor.tensor_type) == 42
    assert get_gguf_weight_type_map(path, {"test.weight": "test.weight"}) == {
        "test.weight": "Q2_0"
    }

    weights = list(gguf_quant_weights_iterator(path, None))
    assert [name for name, _ in weights] == ["test.qweight_type", "test.qweight"]
    assert weights[0][1].item() == 42
    assert torch.equal(weights[1][1], torch.from_numpy(packed))


def test_real_bonsai_tensor_types_are_all_recognized() -> None:
    assert BONSAI_MODEL.is_file(), f"Missing test model: {BONSAI_MODEL}"
    tensors = gguf.GGUFReader(BONSAI_MODEL).tensors
    histogram = Counter(int(tensor.tensor_type) for tensor in tensors)

    assert histogram == Counter({42: 197, 0: 113})
    assert all(
        tensor.tensor_type in DEQUANT_TYPES
        or tensor.tensor_type
        in {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        }
        for tensor in tensors
    )
    token_embedding = next(t for t in tensors if t.name == "token_embd.weight")
    assert token_embedding.tensor_type == Q2_0


def test_ternary_quant_names_are_valid() -> None:
    assert is_valid_gguf_quant_type("Q1_0")
    assert is_valid_gguf_quant_type("Q2_0")


def test_prism_gguf_layout_guard_requirements() -> None:
    assert gguf.GGML_QUANT_SIZES[Q1_0] == (128, 18)
    assert gguf.GGML_QUANT_SIZES[Q2_0] == (128, 34)
    assert Q1_0 in DEQUANT_TYPES
    assert Q2_0 in DEQUANT_TYPES
