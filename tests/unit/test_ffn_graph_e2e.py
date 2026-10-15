# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""The FFN graph E2E must prove real execution, not just HTTP success."""

import copy
from types import SimpleNamespace

import pytest

from tests.e2e.models.deepseek_v4_flash import ffn_graph


def evidence(mode) -> tuple[dict, dict]:
    before: dict = {"ffn": {"results": []}, "attention": {"results": []}}
    after: dict = {"attention": {"results": []}, "ffn": {"results": []}}
    for rank in range(4):
        before["attention"]["results"].append(
            {"rank": rank, "attention": {"phases": {"live": 0}}}
        )
        after["attention"]["results"].append(
            {
                "rank": rank,
                "runner": "AFDNPUAttentionModelRunnerV2",
                "attention": {"phases": {"live": 8}},
            }
        )
    for rank in range(4, 8):
        before["ffn"]["results"].append(
            {
                "rank": rank,
                "completion": {"layers": {"0": 2}},
                "replay_count": 2,
            }
        )
        after["ffn"]["results"].append(
            {
                "rank": rank,
                "runner": "AsyncNPUFFNModelRunner",
                "mode": mode,
                "layer_ids": list(range(43)),
                "capture_count": 1,
                "replay_count": 100,
                "completion": {"layers": {str(i): 3 for i in range(43)}},
            }
        )
    return before, after


@pytest.mark.parametrize("mode", ["eager", "full"])
def test_execution_evidence(mode):
    ffn_graph.audit_execution(*evidence(mode), mode)


@pytest.mark.parametrize(
    "failure",
    ["mrv1", "fallback", "startup-only", "missing-layer", "missing-rank", "recapture"],
)
def test_execution_audit_rejects_incomplete_runs(failure):
    before, after = copy.deepcopy(evidence("full"))
    ffn = after["ffn"]["results"][0]
    if failure == "mrv1":
        after["attention"]["results"][0]["runner"] = "MRV1"
    elif failure == "fallback":
        ffn["mode"] = "eager"
    elif failure == "startup-only":
        ffn["replay_count"] = 2
    elif failure == "missing-layer":
        ffn["layer_ids"].pop()
    elif failure == "missing-rank":
        after["ffn"]["results"].pop()
    else:
        ffn["capture_count"] = 2
    with pytest.raises(AssertionError):
        ffn_graph.audit_execution(before, after, "full")


def test_full_batch_samples_do_not_require_every_layer():
    before, after = evidence("full")
    for record in after["ffn"]["results"]:
        record["completion"]["layers"] = {"42": 3}
    ffn_graph.audit_execution(before, after, "full")


def test_eager_still_requires_each_observed_layer_to_advance():
    before, after = evidence("eager")
    after["ffn"]["results"][0]["completion"]["layers"]["0"] = 2
    with pytest.raises(AssertionError, match="No live FFN completion"):
        ffn_graph.audit_execution(before, after, "eager")


def test_only_ffn_graph_mode_changes_between_profiles(tmp_path):
    args = SimpleNamespace(
        model="/model",
        vllm_bin="vllm",
        mode="eager",
        devices="0,1,2,3,4,5,6,7",
        afd_host="127.0.0.1",
        afd_port=29880,
        api_port=19280,
        output_dir=tmp_path,
    )
    profile = ffn_graph.profile(args)
    eager = {r: ffn_graph.command_for(args, profile, r) for r in ("attention", "ffn")}
    args.mode = "full"
    full = {r: ffn_graph.command_for(args, profile, r) for r in ("attention", "ffn")}
    assert eager["attention"] == full["attention"]
    assert "--enforce-eager" in full["attention"]
    full["ffn"].remove("--compilation-config")
    full["ffn"].remove('{"cudagraph_mode":"FULL"}')
    eager["ffn"].remove("--enforce-eager")
    assert full["ffn"] == eager["ffn"]
    for role in ("attention", "ffn"):
        env = ffn_graph.environment_for(args, profile, role)
        assert env["VLLM_USE_V2_MODEL_RUNNER"] == "1"
        assert env["AFD_ASYNC_CAM_LAYERED_GMM"] == ("1" if role == "ffn" else "0")
