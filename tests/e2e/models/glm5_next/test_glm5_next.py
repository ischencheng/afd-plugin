# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CUDA GLM-5.3-Flash E2E scenarios.

The official FP8 checkpoint is about 328 GB, of which the dense MLP and MoE
layers owned by FFN take about 306 GB. With two FFN ranks each FFN GPU holds
about 153 GB of weights, so these scenarios need four GPUs with at least
180 GB of memory each (for example B200).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.conftest import (
    download_dataset,
    download_model,
    preserve_environment_variable,
    run_runner,
)

GSM8K_DATASET_ID = "openai/gsm8k"
GSM8K_DATASET_CONFIG = "main"
GLM5_NEXT_REPO_ID = "zai-org/GLM-5.3-Flash"
GLM5_NEXT_MAX_MODEL_LEN = 4096
# vLLM v0.30.0 launches the KDA recurrent decode kernel with grid z equal to
# sequences x KDA heads (64 per TP1 rank), and CUDA caps grid z at 65535. The
# OpenAI server default of 1024 sequences on large GPUs exceeds it in eager mode.
GLM5_NEXT_MAX_NUM_SEQS = 512
DEFAULT_DEVICE_IDS = ("0", "1", "2", "3")
ATTENTION_DEVICE_COUNT = 2
AFD_FFN_DEVICE_COUNT = 2
BASELINE_DEVICE_COUNT = 4
# Native MRV2 does not support DBO for hybrid KDA models.
SCENARIOS = (
    "baseline-graph",
    "afd-eager-2a2f",
    "afd-graph-2a2f",
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def prepare_e2e_assets() -> None:
    """Ensure GSM8K and GLM-5.3-Flash are available for the runner."""
    if _required_env("AFD_E2E_BACKEND") != "gpu":
        raise RuntimeError("GLM-5.3-Flash E2E supports only the 'gpu' backend")

    download_dataset(GSM8K_DATASET_ID, GSM8K_DATASET_CONFIG)

    existing = os.environ.get("AFD_GPU_E2E_MODEL")
    if existing:
        print(
            f"[e2e] Using existing model path {Path(existing).expanduser()}",
            flush=True,
        )
        return

    os.environ["AFD_GPU_E2E_MODEL"] = str(download_model(GLM5_NEXT_REPO_ID))


def build_runner_command(scenario: str, gsm8k_output_path: Path) -> list[str]:
    backend = _required_env("AFD_E2E_BACKEND")
    if backend != "gpu":
        raise RuntimeError("GLM-5.3-Flash E2E supports only the 'gpu' backend")

    env_devices = os.environ.get("AFD_E2E_DEVICES")
    devices = (
        [item.strip() for item in env_devices.split(",") if item.strip()]
        if env_devices
        else list(DEFAULT_DEVICE_IDS)
    )
    if scenario == "baseline-graph":
        attention_devices = devices[:BASELINE_DEVICE_COUNT]
        ffn_devices = []
    else:
        attention_devices = devices[:ATTENTION_DEVICE_COUNT]
        ffn_devices = devices[
            ATTENTION_DEVICE_COUNT : ATTENTION_DEVICE_COUNT + AFD_FFN_DEVICE_COUNT
        ]

    command = [
        sys.executable,
        "-m",
        "tests.e2e.runner",
        "--model",
        _required_env("AFD_GPU_E2E_MODEL"),
        "--vllm-bin",
        os.environ.get("AFD_GPU_E2E_VLLM_BIN", "vllm"),
        "--device-backend",
        backend,
        f"--common-vllm-arg=--max-model-len={GLM5_NEXT_MAX_MODEL_LEN}",
        "--common-vllm-arg=--language-model-only",
        "--attention-devices",
        ",".join(attention_devices),
    ]
    if scenario == "afd-eager-2a2f":
        command.append(f"--common-vllm-arg=--max-num-seqs={GLM5_NEXT_MAX_NUM_SEQS}")
    if scenario != "baseline-graph":
        command.extend(["--ffn-devices", ",".join(ffn_devices)])
    command.extend(
        [
            "--scenario",
            scenario,
            "--gsm8k-output-path",
            str(gsm8k_output_path),
            "--served-model-name-prefix",
            "glm5-next-afd",
        ],
    )
    return command


@pytest.fixture(scope="module", autouse=True)
def _prepare_e2e_assets() -> Iterator[None]:
    """Prepare shared E2E assets once for this test module."""
    with preserve_environment_variable("AFD_GPU_E2E_MODEL"):
        prepare_e2e_assets()
        yield


@pytest.mark.e2e
@pytest.mark.parametrize("scenario", SCENARIOS, ids=SCENARIOS)
def test_glm5_next(scenario: str, tmp_path: Path) -> None:
    command = build_runner_command(scenario, tmp_path / scenario)
    run_runner(command)
