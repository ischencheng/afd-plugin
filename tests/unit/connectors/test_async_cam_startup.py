# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pure host negotiation/drain tests; no device or model runner initialization."""

import ast
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Condition, Event
from types import ModuleType, SimpleNamespace
from typing import Literal

import pytest

# Compile the real connector's host-only methods without importing its NPU
# dependencies. This keeps the existing CPU negotiation/drain tests runnable;
# full connector construction is covered by test_async_ffn_runtime.py.
source = Path(__file__).resolve().parents[3] / "afd_plugin/connectors/npu/async_cam.py"
module_ast = ast.parse(source.read_text())
host_methods = {
    "_graph_keys",
    "_graph_publish",
    "check_graph_failure",
    "fail_graph_startup",
    "_graph_wait",
    "wait_graph_device",
    "graph_phase",
    "_negotiate_graph",
    "graph_service_ready",
    "_wait_graph_service_ready",
    "_all_attention_drained",
    "_graph_ffn_stopped",
    "_finish_attention_graph_shutdown",
}
body: list[ast.stmt] = [
    ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
]
for node in module_ast.body:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in (
        "FFNGraphPlan",
        "resolve_ffn_mode",
        "CAMAsyncAFDConnector",
    ):
        if isinstance(node, ast.ClassDef) and node.name == "CAMAsyncAFDConnector":
            node.bases = []
            node.body = [
                m
                for m in node.body
                if isinstance(m, ast.FunctionDef) and m.name in host_methods
            ]
            assert {
                m.name for m in node.body if isinstance(m, ast.FunctionDef)
            } == host_methods
        body.append(node)
    elif isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"SHUTDOWN_TIMEOUT_SECONDS", "POLL_SECONDS"}
        for t in node.targets
    ):
        body.append(node)
startup_module = ModuleType("afd_graph_startup_contract")
sys.modules[startup_module.__name__] = startup_module
vars(startup_module).update(
    json=json, time=time, asdict=asdict, dataclass=dataclass, Literal=Literal
)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
        str(source),
        "exec",
    ),
    vars(startup_module),
)
CAMAsyncAFDConnector = startup_module.CAMAsyncAFDConnector
FFNGraphPlan = startup_module.FFNGraphPlan
resolve_ffn_mode = startup_module.resolve_ffn_mode

from afd_plugin.validation import (  # noqa: E402
    cam_receive_capacity,
    requested_ffn_mode,
    validate_graph_capacity,
)


class MemoryStore:
    def __init__(self):
        self.values = {}
        self.condition = Condition()
        self.closed = False

    def set(self, key, value):
        assert not self.closed, "Store used after release"
        with self.condition:
            self.values[key] = value.encode() if isinstance(value, str) else value
            self.condition.notify_all()

    def get(self, key):
        assert not self.closed, "Store used after release"
        with self.condition:
            assert self.condition.wait_for(lambda: key in self.values, timeout=3)
            return self.values[key]

    def check(self, keys):
        assert not self.closed, "Store used after release"
        with self.condition:
            return all(key in self.values for key in keys)


def startup_world(attn=2, ffn=2, store=None):
    store = store or MemoryStore()
    owners = []
    for rank in range(attn + ffn):
        owner = CAMAsyncAFDConnector()
        owner._graph_store = store
        owner.world_rank, owner.attn_size = rank, attn
        owner.topology = SimpleNamespace(world_size=attn + ffn)
        owner._graph_prefix = "ffn_graph/test/"
        owner.ffn_graph_mode = "eager"
        owner.ffn_graph_fallback_reason = ""
        owner._graph_warmup_layer_id = None
        owner._graph_attention_finished = False
        owners.append(owner)
    return owners


def full_plan(**changes):
    return replace(
        FFNGraphPlan(requested="full", signature="weights/capacity", layer_ids=(2, 5)),
        **changes,
    )


@pytest.mark.parametrize("mode", ["FULL_DECODE_ONLY", "PIECEWISE", "NONE"])
def test_full_requires_explicit_complete_transaction_mode(mode):
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name=mode)),
    )
    with pytest.raises(ValueError, match="explicit"):
        requested_ffn_mode(config)
    config.model_config.enforce_eager = True
    assert requested_ffn_mode(config) == "eager"
    config.model_config.enforce_eager = False
    config.compilation_config.cudagraph_mode.name = "FULL"
    assert requested_ffn_mode(config) == "full"


def test_mixed_requested_modes_are_not_silently_downgraded():
    with pytest.raises(ValueError, match="same execution mode"):
        resolve_ffn_mode([full_plan(allow_eager_fallback=True), FFNGraphPlan()])


def test_unsupported_rank_requires_every_rank_to_explicitly_allow_eager():
    unsupported = full_plan(
        graph_error="unsupported executor", allow_eager_fallback=True
    )
    with pytest.raises(ValueError, match="unsupported executor"):
        resolve_ffn_mode([full_plan(), unsupported])
    assert resolve_ffn_mode([full_plan(allow_eager_fallback=True), unsupported]) == (
        "eager",
        "unsupported executor",
    )


def test_layout_mismatch_is_a_static_full_rejection():
    with pytest.raises(ValueError, match="signatures differ"):
        resolve_ffn_mode([full_plan(), full_plan(signature="other-capacity")])
    with pytest.raises(ValueError, match="layer tables"):
        resolve_ffn_mode([full_plan(), full_plan(layer_ids=(2, 6))])
    assert resolve_ffn_mode([FFNGraphPlan(graph_error="not graph capable")]) == (
        "eager",
        "",
    )


def test_empty_graph_signature_cannot_advertise_support():
    with pytest.raises(ValueError, match="loaded executor"):
        resolve_ffn_mode([full_plan(signature="")])


def test_whole_world_agrees_before_service_readiness():
    owners = startup_world()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                owner._negotiate_graph, None if i < 2 else full_plan(), "same-wire"
            )
            for i, owner in enumerate(owners)
        ]
        for future in futures:
            future.result(timeout=3)
    assert all(
        owner.ffn_graph_mode == "full" and owner._graph_warmup_layer_id == 2
        for owner in owners
    )
    for owner in owners[2:]:
        owner.graph_service_ready()
    for owner in owners[:2]:
        owner._wait_graph_service_ready()


@pytest.mark.parametrize(
    "plan",
    [
        full_plan(),
        full_plan(graph_error="graph DR unavailable", allow_eager_fallback=True),
    ],
    ids=["strict-full", "full-with-eager-fallback"],
)
def test_transport_mismatch_rejects_before_payload_initialization(plan):
    owners = startup_world(1, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(owner._negotiate_graph, None if i == 0 else plan, str(i))
            for i, owner in enumerate(owners)
        ]
        for future in futures:
            with pytest.raises(
                (ValueError, RuntimeError), match="transport signatures"
            ):
                future.result(timeout=3)


@pytest.mark.parametrize(
    "plan,transports,reason",
    [
        (
            full_plan(graph_error="graph DR unavailable", allow_eager_fallback=True),
            ("same-wire", "same-wire"),
            "graph DR unavailable",
        ),
        (FFNGraphPlan(), ("wire-a", "wire-b"), ""),
    ],
    ids=["matching-transport-fallback", "ordinary-eager-unchanged"],
)
def test_eager_negotiation_preserves_transport_check_scope(plan, transports, reason):
    owners = startup_world(1, 1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(owner._negotiate_graph, None if i == 0 else plan, transports[i])
            for i, owner in enumerate(owners)
        ]
        for future in futures:
            future.result(timeout=3)
    assert all(
        owner.ffn_graph_mode == "eager"
        and owner.ffn_graph_fallback_reason == reason
        and owner._graph_warmup_layer_id is None
        for owner in owners
    )


def test_capacity_matches_native_float32_factor(monkeypatch):
    monkeypatch.setenv("BATCH_SIZE_FACTOR", "0.000061035156249")
    # Python float truncates to 15; native std::stof rounds this to 16 rows.
    assert cam_receive_capacity() == 16


@pytest.mark.parametrize(
    "tokens,tp,capacity", [(0, 1, 16), (7, 2, 16), (32, 1, 16), (65536, 1, 65536)]
)
def test_graph_rejects_unsafe_static_envelope(tokens, tp, capacity):
    with pytest.raises(ValueError):
        validate_graph_capacity(tokens, tp, capacity)


def test_graph_accepts_conservative_uint16_boundary():
    validate_graph_capacity(65535, 1, 65535)
    validate_graph_capacity(65536, 2, 65536)


def test_attention_keeps_store_until_all_ffn_and_attention_are_finished():
    nonowner_at_final_write = Event()
    allow_nonowner_final_write = Event()

    class DelayedStore(MemoryStore):
        def set(self, key, value):
            if key.endswith("finished/1"):
                nonowner_at_final_write.set()
                assert allow_nonowner_final_write.wait(3)
            super().set(key, value)

    owners = startup_world(store=DelayedStore())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(owner._finish_attention_graph_shutdown) for owner in owners[:2]
        ]
        store = owners[0]._graph_store
        with store.condition:
            assert store.condition.wait_for(owners[2]._all_attention_drained, timeout=3)
        assert not any(future.done() for future in futures)
        for owner in owners[2:]:
            owner._graph_ffn_stopped()
        assert nonowner_at_final_write.wait(3)
        assert not futures[0].done()
        allow_nonowner_final_write.set()
        for future in futures:
            future.result(timeout=3)
    store.closed = True
    for owner in owners[:2]:
        # Repeated shutdown never touches Store.
        owner._finish_attention_graph_shutdown()
