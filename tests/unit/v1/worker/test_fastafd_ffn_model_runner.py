# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.connectors import AFDDPMetadata  # noqa: E402
from afd_plugin.connectors.gpu.fastafd import M2NShape  # noqa: E402
from afd_plugin.v1.worker import fastafd_ffn_model_runner as runner_module  # noqa: E402
from afd_plugin.v1.worker.fastafd_ffn_model_runner import (  # noqa: E402
    FastAFDFFNModelRunner,
)

_SHAPE = M2NShape(
    num_attention_ranks=2,
    num_ffn_ranks=2,
    num_experts=256,
    top_k=8,
    hidden_size=2048,
    intermediate_size=512,
    max_tokens_per_rank=1024,
)


def _lane(name):
    return SimpleNamespace(
        buffer=f"{name}.buffer",
        peer_ptrs=[f"{name}.peer"],
        max_tokens_per_rank=1152,
    )


def _runner(monkeypatch, *, num_lanes=2):
    launches = []
    monkeypatch.setattr(
        runner_module,
        "fastafd_ops",
        lambda: SimpleNamespace(mega_moe_m2n_eg=lambda *args: launches.append(args)),
    )
    runner = object.__new__(FastAFDFFNModelRunner)
    runner.prof = None
    runner.connector = SimpleNamespace(
        shape=_SHAPE,
        union_rank=3,
        lanes=[_lane(f"lane{i}") for i in range(num_lanes)],
    )
    runner.expert_weights = SimpleNamespace(
        descriptors="descriptors",
        l1_pointers="l1_pointers",
        num_layers=40,
        l1_nbytes=4096,
    )
    return runner, launches


def _plan(*stage_tokens):
    return {
        stage: AFDDPMetadata(num_tokens_across_dp_cpu=tokens)
        for stage, tokens in enumerate(stage_tokens)
    }


def test_one_launch_serves_every_layer_and_lane(monkeypatch):
    runner, launches = _runner(monkeypatch)

    runner.submit_step(_plan([3, 9], [5, 0]))

    assert len(launches) == 1
    args = launches[0]
    assert args[:7] == ("descriptors", "l1_pointers", 40, 128, 2048, 512, 4096)
    assert args[9:13] == ("lane0.buffer", "lane1.buffer", None, None)
    assert args[13:17] == (
        ["lane0.peer"],
        ["lane1.peer"],
        ["lane0.peer"],
        ["lane0.peer"],
    )
    # union rank, M/N ranks, aligned capacity, experts, top-k, expected tokens.
    assert args[17:24] == (3, 2, 2, 1152, 256, 8, 9)


def test_single_stage_step_uses_only_the_first_lane(monkeypatch):
    runner, launches = _runner(monkeypatch)

    runner.submit_step(_plan([0, 0]))

    args = launches[0]
    assert args[9:13] == ("lane0.buffer", None, None, None)
    assert args[23] == 1


def test_graph_capture_steps_do_not_launch_the_expert_kernel(monkeypatch):
    runner, launches = _runner(monkeypatch)

    runner.execute_model(dp_metadata_list=_plan([4, 4]), is_graph_capturing=True)
    assert launches == []

    runner.execute_model(dp_metadata_list=_plan([4, 4]))
    assert len(launches) == 1


def test_capture_model_serves_attention_warmup_only(monkeypatch):
    runner, launches = _runner(monkeypatch)

    assert runner.capture_model(_plan([4, 4]), is_warmup=False) == 0
    assert launches == []
    assert runner.capture_model(_plan([4, 4]), is_warmup=True) == 0
    assert len(launches) == 1


def test_steps_require_a_plan(monkeypatch):
    runner, _launches = _runner(monkeypatch)

    with pytest.raises(RuntimeError, match="requires dp_metadata_list"):
        runner.execute_model()


def test_dummy_weights_are_rejected():
    runner = object.__new__(FastAFDFFNModelRunner)

    with pytest.raises(ValueError, match="needs checkpoint expert weights"):
        runner.load_model(load_dummy_weights=True)
