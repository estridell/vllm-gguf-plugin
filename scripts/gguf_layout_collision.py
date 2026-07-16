#!/usr/bin/env python3
"""Generate and inspect minimal GGUF fixtures for the Q1_0/Q2_0 collision."""

from __future__ import annotations

import argparse
import hashlib
import struct
from pathlib import Path
from typing import NamedTuple

ALIGNMENT = 32
FIXTURE_DATA_BYTES = 64
Q1_0_TYPE_ID = 41
Q2_0_TYPE_ID = 42


class Fixture(NamedTuple):
    filename: str
    type_id: int
    payload: bytes
    logical_payload_bytes: int


def pack_q1_0(scale: float, signs: tuple[bool, ...]) -> bytes:
    if len(signs) != 128:
        raise ValueError("Q1_0 requires exactly 128 signs")
    packed = bytearray(16)
    for index, positive in enumerate(signs):
        packed[index // 8] |= int(positive) << (index % 8)
    return struct.pack("<e", scale) + packed


def pack_q2_0(scale: float, codes: tuple[int, ...], block_size: int) -> bytes:
    if block_size not in (64, 128) or len(codes) != block_size:
        raise ValueError("Q2_0 requires exactly 64 or 128 codes")
    if any(code < 0 or code > 3 for code in codes):
        raise ValueError("Q2_0 codes must be in [0, 3]")
    packed = bytearray(block_size // 4)
    for index, code in enumerate(codes):
        packed[index // 4] |= code << (2 * (index % 4))
    return struct.pack("<e", scale) + packed


def dequantize_q2_0(payload: bytes, elements: int, block_size: int) -> list[float]:
    block_bytes = 2 + block_size // 4
    if elements % block_size:
        raise ValueError("element count must be divisible by the block size")
    values: list[float] = []
    for block_index in range(elements // block_size):
        start = block_index * block_bytes
        block = payload[start : start + block_bytes]
        if len(block) != block_bytes:
            raise ValueError("payload is too short for the requested interpretation")
        scale = struct.unpack("<e", block[:2])[0]
        for index in range(block_size):
            code = (block[2 + index // 4] >> (2 * (index % 4))) & 0x03
            values.append((code - 1) * scale)
    return values


def build_gguf(type_id: int, payload: bytes, elements: int = 128) -> bytes:
    """Build a GGUF v3 file with one 1-D tensor and no metadata."""
    if len(payload) > FIXTURE_DATA_BYTES:
        raise ValueError("fixture payload exceeds its fixed data allocation")
    name = b"test.weight"
    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 0)
    tensor_info = (
        struct.pack("<Q", len(name))
        + name
        + struct.pack("<I", 1)
        + struct.pack("<Q", elements)
        + struct.pack("<I", type_id)
        + struct.pack("<Q", 0)
    )
    directory = header + tensor_info
    data_offset = (len(directory) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    data = payload + bytes(FIXTURE_DATA_BYTES - len(payload))
    return directory + bytes(data_offset - len(directory)) + data


def make_fixtures() -> tuple[Fixture, ...]:
    repeating_codes_64 = tuple(range(4)) * 16
    repeating_codes_128 = tuple(range(4)) * 32
    q1_payload = pack_q1_0(1.0, tuple(index % 2 == 1 for index in range(128)))
    upstream_q2_payload = pack_q2_0(1.0, repeating_codes_64, 64) + pack_q2_0(
        2.0, repeating_codes_64, 64
    )
    prism_q2_payload = pack_q2_0(1.0, repeating_codes_128, 128)
    return (
        Fixture("q1_0-id41.gguf", Q1_0_TYPE_ID, q1_payload, len(q1_payload)),
        Fixture(
            "q2_0-id42-upstream.gguf",
            Q2_0_TYPE_ID,
            upstream_q2_payload,
            len(upstream_q2_payload),
        ),
        Fixture(
            "q2_0-id42-prism.gguf",
            Q2_0_TYPE_ID,
            prism_q2_payload,
            len(prism_q2_payload),
        ),
    )


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def render_report(fixtures: tuple[Fixture, ...]) -> str:
    files = {
        fixture.filename: build_gguf(fixture.type_id, fixture.payload)
        for fixture in fixtures
    }
    upstream_fixture = fixtures[1]
    prism_fixture = fixtures[2]
    upstream_file = files[upstream_fixture.filename]
    prism_file = files[prism_fixture.filename]
    data_offset = len(upstream_file) - FIXTURE_DATA_BYTES

    upstream_as_prism = dequantize_q2_0(upstream_fixture.payload, 128, 128)
    prism_as_upstream = dequantize_q2_0(prism_fixture.payload + bytes(2), 128, 64)

    lines = [
        "Q1_0/Q2_0 GGUF layout collision reproducer",
        f"data offset: {data_offset}",
    ]
    for fixture in fixtures:
        file_data = files[fixture.filename]
        lines.append(
            f"{fixture.filename}: type={fixture.type_id}, elements=128, "
            f"logical_payload={fixture.logical_payload_bytes}, "
            f"file_bytes={len(file_data)}, sha256={sha256(file_data)}"
        )
    lines.extend(
        (
            "ID-42 tensor directories equal: "
            + str(upstream_file[:data_offset] == prism_file[:data_offset]),
            f"ID-42 total file lengths equal: {len(upstream_file) == len(prism_file)}",
            "upstream payload: " + upstream_fixture.payload.hex(" "),
            "Prism payload:   " + prism_fixture.payload.hex(" "),
            "upstream decoded as Prism, values[64:72]: "
            + repr(upstream_as_prism[64:72]),
            "Prism decoded as upstream, values[64:72]: "
            + repr(prism_as_upstream[64:72]),
        )
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="write the three deterministic GGUF fixtures to this directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fixtures = make_fixtures()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for fixture in fixtures:
            (args.output_dir / fixture.filename).write_bytes(
                build_gguf(fixture.type_id, fixture.payload)
            )
    print(render_report(fixtures))


if __name__ == "__main__":
    main()
