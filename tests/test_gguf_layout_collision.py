from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "gguf_layout_collision.py"
SPEC = importlib.util.spec_from_file_location("gguf_layout_collision", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
layout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(layout)


def test_q1_0_fixture_is_the_shared_18_byte_layout() -> None:
    fixture = layout.make_fixtures()[0]

    assert fixture.type_id == 41
    assert fixture.logical_payload_bytes == 18
    assert fixture.payload == bytes.fromhex("00 3c" + " aa" * 16)


def test_q2_0_fixtures_have_identical_directories_and_file_lengths() -> None:
    _, upstream, prism = layout.make_fixtures()
    upstream_file = layout.build_gguf(upstream.type_id, upstream.payload)
    prism_file = layout.build_gguf(prism.type_id, prism.payload)
    data_offset = len(upstream_file) - layout.FIXTURE_DATA_BYTES

    assert upstream.type_id == prism.type_id == 42
    assert upstream.logical_payload_bytes == 36
    assert prism.logical_payload_bytes == 34
    assert upstream_file[:data_offset] == prism_file[:data_offset]
    assert len(upstream_file) == len(prism_file) == 160
    assert upstream_file[data_offset:] != prism_file[data_offset:]


def test_wrong_q2_0_interpretation_silently_changes_values() -> None:
    _, upstream, prism = layout.make_fixtures()

    upstream_correct = layout.dequantize_q2_0(upstream.payload, 128, 64)
    upstream_as_prism = layout.dequantize_q2_0(upstream.payload, 128, 128)
    prism_correct = layout.dequantize_q2_0(prism.payload, 128, 128)
    prism_as_upstream = layout.dequantize_q2_0(prism.payload + bytes(2), 128, 64)

    assert upstream_correct[64:72] == [-2.0, 0.0, 2.0, 4.0] * 2
    assert upstream_as_prism[64:72] == [-1.0] * 7 + [0.0]
    assert prism_correct[64:72] == [-1.0, 0.0, 1.0, 2.0] * 2
    assert prism_as_upstream[64:72] == [1252.0, 0.0, -1252.0, -2504.0] * 2
