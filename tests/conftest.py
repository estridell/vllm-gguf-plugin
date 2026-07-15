# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ternary integration tests")
    group.addoption(
        "--ternary-model",
        default=os.environ.get("VLLM_GGUF_TEST_TERNARY_MODEL"),
        metavar="PATH",
        help=(
            "Path to a full ternary GGUF model. Defaults to "
            "VLLM_GGUF_TEST_TERNARY_MODEL."
        ),
    )
    group.addoption(
        "--ternary-4b-model",
        default=os.environ.get("VLLM_GGUF_TEST_TERNARY_4B_MODEL"),
        metavar="PATH",
        help=(
            "Path to the Bonsai 4B GGUF model. Defaults to "
            "VLLM_GGUF_TEST_TERNARY_4B_MODEL."
        ),
    )
    group.addoption(
        "--ternary-4b-config",
        default=os.environ.get("VLLM_GGUF_TEST_TERNARY_4B_CONFIG"),
        metavar="PATH",
        help=(
            "Path to the Bonsai 4B Hugging Face config. Defaults to "
            "VLLM_GGUF_TEST_TERNARY_4B_CONFIG."
        ),
    )


def _required_external_path(
    pytestconfig: pytest.Config,
    option: str,
    environment_variable: str,
    *,
    directory: bool = False,
) -> Path:
    configured_path = pytestconfig.getoption(option)
    if not configured_path:
        pytest.skip(f"set {option} or {environment_variable} to run this test")

    path = Path(configured_path).expanduser()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        expected = "directory" if directory else "file"
        pytest.skip(f"configured {expected} does not exist: {path}")
    return path


@pytest.fixture(scope="session")
def ternary_model_path(pytestconfig: pytest.Config) -> Path:
    return _required_external_path(
        pytestconfig,
        "--ternary-model",
        "VLLM_GGUF_TEST_TERNARY_MODEL",
    )


@pytest.fixture(scope="session")
def ternary_4b_model_path(pytestconfig: pytest.Config) -> Path:
    return _required_external_path(
        pytestconfig,
        "--ternary-4b-model",
        "VLLM_GGUF_TEST_TERNARY_4B_MODEL",
    )


@pytest.fixture(scope="session")
def ternary_4b_config_path(pytestconfig: pytest.Config) -> Path:
    return _required_external_path(
        pytestconfig,
        "--ternary-4b-config",
        "VLLM_GGUF_TEST_TERNARY_4B_CONFIG",
        directory=True,
    )


@pytest.fixture
def example_prompts() -> list[str]:
    return [
        (
            "vLLM is a high-throughput and memory-efficient inference and "
            "serving engine for LLMs."
        ),
        "Briefly describe the major phases of the moon.",
        "Explain the concept of artificial intelligence in simple terms.",
        "What are the main differences between Python and C++?",
    ]
