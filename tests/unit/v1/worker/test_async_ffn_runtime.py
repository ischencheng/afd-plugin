# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""MRV2 loading, eager isolation and graph-only resource lifetime contracts."""

from __future__ import annotations

import sys
import threading
from collections import deque
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
import torch  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors.metadata import (  # noqa: E402
    AFDA2FTransferPayload,
    AFDTransferContext,
    AFDTransferMetadata,
)
from afd_plugin.connectors.npu.async_cam import (  # noqa: E402
    AFDAsyncTransferState,
    CAMAsyncAFDConnector,
)
from afd_plugin.v1.worker.npu import async_ffn_model_runner as runtime  # noqa: E402
from afd_plugin.v1.worker.npu.ffn_worker import AFDNPUFFNWorker  # noqa: E402


def config():
    return SimpleNamespace(
        use_v2_model_runner=True,
        additional_config={
            "afd": {
                "role": "ffn",
                "connector": "CAMAsyncAFDConnector",
                "num_attention_ranks": 2,
                "num_ffn_ranks": 2,
                "compute_gate_on_attention": True,
                "async": True,
            }
        },
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                hidden_size=32, num_experts_per_tok=2, n_routed_experts=4
            ),
            dtype=torch.bfloat16,
            enforce_eager=True,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        parallel_config=SimpleNamespace(enable_dbo=False, use_ubatching=False),
    )


@pytest.fixture
def harness(monkeypatch):
    cfg = config()
    connector = CAMAsyncAFDConnector(
        2,
        0,
        cfg,
        AFDConfig(
            role="ffn",
            connector="CAMAsyncAFDConnector",
            num_attention_ranks=2,
            num_ffn_ranks=2,
            compute_gate_on_attention=True,
            async_dp=True,
        ),
        0,
    )
    connector._initialized = True
    monkeypatch.setenv("AFD_ASYNC_CAM_LAYERED_GMM", "0")
    monkeypatch.setattr(runtime, "create_afd_npu_profiler", lambda role: None)
    runner = runtime.AsyncNPUFFNModelRunner(cfg, torch.device("cpu"), connector)
    runner.layer_ids = (2, 5)
    calls: list[str | tuple] = []
    payloads: deque[AFDA2FTransferPayload] = deque()

    def receive(**kwargs):
        calls.append("receive")
        return payloads.popleft()

    def compute(**kwargs):
        calls.append(("compute", kwargs["layer_idx"], kwargs["hidden_states"].shape[0]))
        return kwargs["hidden_states"].to(torch.bfloat16)

    def send(output, context, **kwargs):
        calls.append(("send", context.states.token_nums_rankid_layeridx))

    @contextmanager
    def forward(**kwargs):
        yield SimpleNamespace(dp_metadata=None, additional_kwargs={})

    monkeypatch.setattr(connector, "recv_attn_output", receive)
    monkeypatch.setattr(connector, "send_ffn_output", send)
    monkeypatch.setattr(runtime, "ascend_forward_context", forward)
    monkeypatch.setattr(runtime, "_set_moe_layer_index", lambda *args: None)
    runner.model = SimpleNamespace(compute_ffn_output=compute)
    return SimpleNamespace(
        runner=runner, connector=connector, calls=calls, payloads=payloads
    )


def payload(layer, rows):
    state = AFDAsyncTransferState(
        token_nums_rankid_layeridx=torch.tensor([rows, 0, layer, 0, 1]),
        group_list=torch.tensor([rows, 0]),
        dynamic_scales=torch.ones(8),
    )
    return AFDA2FTransferPayload(
        torch.ones(8, 32, dtype=torch.int8),
        AFDTransferContext(AFDTransferMetadata(0, 0, [8]), state),
    )


def test_eager_uses_main_layer_loop_and_original_return_headers(harness):
    h = harness
    first, zero = payload(5, 3), payload(2, 0)
    h.payloads.extend((first, zero))
    assert h.runner.execute_connector_driven_step()
    assert [
        call for call in h.calls if isinstance(call, tuple) and call[0] == "compute"
    ] == [("compute", 5, 3), ("compute", 2, 0)]
    returned = [
        call[1] for call in h.calls if isinstance(call, tuple) and call[0] == "send"
    ]
    assert returned[0] is first.context.states.token_nums_rankid_layeridx
    assert returned[1] is zero.context.states.token_nums_rankid_layeridx
    assert not h.connector.graph_receiving and h.connector._graph_stop is None


def test_layered_eager_uses_raw_device_metadata(harness):
    h = harness
    h.payloads.extend((payload(5, 3), payload(2, 0)))
    observed = []

    def layered(hidden, scales, groups, header):
        observed.append((groups, header))
        return hidden.to(torch.bfloat16)

    h.runner._layered_executor = layered
    assert h.runner.execute_connector_driven_step()
    assert len(observed) == 2
    assert [int(header[2]) for _, header in observed] == [5, 2]
    assert not any(isinstance(call, tuple) and call[0] == "compute" for call in h.calls)


@pytest.mark.parametrize("speculative_tokens", [None, 3])
def test_direct_loader_uses_role_model_config_and_keeps_memory_accounting(
    harness, monkeypatch, speculative_tokens
):
    h = harness
    model = h.runner.model
    h.runner.model = None
    h.runner.vllm_config.load_config = object()
    h.runner.vllm_config.scheduler_config.max_num_seqs = 6
    h.runner.vllm_config.speculative_config = (
        None
        if speculative_tokens is None
        else SimpleNamespace(num_speculative_tokens=speculative_tokens)
    )
    seen = []
    preparation: list[tuple] = []

    @contextmanager
    def memory_profiler():
        yield SimpleNamespace(consumed_memory=1234)

    def get_loader(load_config):
        assert load_config is h.runner.vllm_config.load_config
        assert preparation == [
            (h.runner.vllm_config, 6, 1 + (speculative_tokens or 0)),
            (h.runner.vllm_config, h.runner.device),
        ]

        def load_model(**kwargs):
            seen.append(kwargs)
            return model

        return SimpleNamespace(load_model=load_model)

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.ascend_forward_context",
        SimpleNamespace(
            set_mc2_tokens_capacity=lambda *args: preparation.append(args),
            set_mc2_mask=lambda *args: preparation.append(args),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader",
        SimpleNamespace(get_model_loader=get_loader),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.utils.mem_utils",
        SimpleNamespace(DeviceMemoryProfiler=memory_profiler),
    )
    h.runner.load_model()
    assert h.runner.get_model() is model
    assert h.runner.model_memory_usage == 1234
    assert seen == [
        {"vllm_config": h.runner.vllm_config, "model_config": h.runner.model_config}
    ]
    with pytest.raises(RuntimeError, match="already loaded"):
        h.runner.load_model()


def test_graph_stop_fences_dma_before_final_store_access(harness):
    c, calls = harness.connector, []
    c._graph_stop_requested = True
    c._stop_done = SimpleNamespace(synchronize=lambda: calls.append("dma-fence"))
    c._graph_ffn_stopped = lambda: calls.append("stopped")
    assert not c.finish_graph(-1)
    assert calls == ["dma-fence", "stopped"] and c._graph_stopped
    c._graph_store = None  # A may now destroy Store.
    c.poll_graph_stop()
    with pytest.raises(RuntimeError, match="slot=-2"):
        c.finish_graph(-2)


def test_graph_stop_waits_for_every_attention_rank(harness, monkeypatch):
    c, calls = harness.connector, []
    drained = False
    c._graph_store = object()
    c._all_attention_drained = lambda: drained
    c._graph_stop = torch.tensor([0, -2])
    c._stop_host = torch.tensor([1])
    c._stop_stream = object()
    c._stop_done = SimpleNamespace(record=lambda stream: calls.append("record"))
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    c.poll_graph_stop()
    assert not calls
    drained = True
    c.poll_graph_stop()
    c.poll_graph_stop()
    assert calls == ["record"]
    assert c._graph_stop.tolist() == [1, -2], "STOP must preserve the first error"


def test_terminal_ack_waits_for_stop_publication(harness, monkeypatch):
    c, calls = harness.connector, []
    recording, allow_record = threading.Event(), threading.Event()
    finishing, finished = threading.Event(), threading.Event()
    errors = []

    def record(stream):
        recording.set()
        assert allow_record.wait(5)
        calls.append("record")

    def finish():
        finishing.set()
        try:
            assert not c.finish_graph(-1)
        except Exception as exc:
            errors.append(exc)
        finally:
            finished.set()

    c._graph_store = object()
    c._all_attention_drained = lambda: True
    c._graph_ffn_stopped = lambda: calls.append("ack")
    c._graph_stop = torch.zeros(2, dtype=torch.int64)
    c._stop_host = torch.ones(1, dtype=torch.int64)
    c._stop_stream = object()
    c._stop_done = SimpleNamespace(
        record=record, synchronize=lambda: calls.append("dma-fence")
    )
    monkeypatch.setattr(torch.npu, "stream", lambda stream: nullcontext())
    control = threading.Thread(target=c.poll_graph_stop)
    service = threading.Thread(target=finish)
    control.start()
    try:
        assert recording.wait(5)
        service.start()
        assert finishing.wait(5)
        assert not finished.wait(0.05)
    finally:
        allow_record.set()
        control.join(5)
        if service.ident is not None:
            service.join(5)
    assert not errors and finished.is_set()
    assert calls == ["record", "dma-fence", "ack"]


def test_attention_graph_shutdown_fences_once_and_never_steals_cr(harness, monkeypatch):
    c, calls = harness.connector, []
    c.world_rank = 0
    c._graph_store = object()
    c.ffn_graph_mode = "full"

    def finish():
        calls.append("drained")
        c._graph_attention_finished = True

    c._finish_attention_graph_shutdown = finish
    monkeypatch.setattr(torch.npu, "synchronize", lambda: calls.append("device-fence"))
    c._pending_attention_payloads[0] = [object()]
    with pytest.raises(RuntimeError, match="consume pending CR"):
        c.quiesce_attention()
    assert not calls and c._pending_attention_payloads
    c._pending_attention_payloads.clear()
    c.quiesce_attention()
    c.quiesce_attention()
    assert calls == ["device-fence", "drained"]


def test_eager_shutdown_does_not_use_graph_store_or_device_fence(harness, monkeypatch):
    c = harness.connector
    c.world_rank = 0
    c._graph_store = object()
    c.ffn_graph_mode = "eager"

    def unexpected():
        pytest.fail("graph drain called for ordinary eager")

    monkeypatch.setattr(torch.npu, "synchronize", unexpected)
    c.quiesce_attention()


@pytest.mark.parametrize("alive", [True, False])
def test_graph_stop_rpc_never_releases_live_thread(harness, monkeypatch, alive):
    calls = []
    runner = harness.runner
    runner.effective_mode = "full"
    runner.shutdown = lambda: calls.append("release")
    worker = AFDNPUFFNWorker.__new__(AFDNPUFFNWorker)
    worker.model_runner = runner
    worker._ffn_shutdown_event = SimpleNamespace(set=lambda: calls.append("event"))
    worker._ffn_thread = SimpleNamespace(
        join=lambda **kwargs: calls.append("join"), is_alive=lambda: alive
    )
    worker._ffn_loop_error = None
    monkeypatch.setattr(
        "afd_plugin.v1.worker.npu.ffn_worker.SHUTDOWN_TIMEOUT_SECONDS", 0
    )
    runner.connector._graph_store = (
        None  # Store can already be gone before RPC resumes.
    )
    if alive:
        with pytest.raises(RuntimeError, match="retain live resources"):
            worker.stop_ffn_server_loop()
        assert calls == []
    else:
        worker.stop_ffn_server_loop()
        assert calls == ["release", "event"]


def test_graph_stop_rpc_takes_over_control_polling(harness):
    calls, live = [], True
    worker = AFDNPUFFNWorker.__new__(AFDNPUFFNWorker)
    worker.model_runner = harness.runner
    harness.runner.effective_mode = "full"
    harness.connector.graph_receiving = True
    harness.connector.poll_graph_stop = lambda: calls.append("poll-stop")
    harness.runner.shutdown = lambda: calls.append("release")
    worker._ffn_loop_error = None
    worker._ffn_shutdown_event = threading.Event()

    def join(timeout):
        nonlocal live
        assert calls == ["poll-stop"] and timeout > 0
        calls.append("join")
        live = False

    worker._ffn_thread = SimpleNamespace(join=join, is_alive=lambda: live)
    worker.stop_ffn_server_loop()
    assert calls == ["poll-stop", "join", "poll-stop", "release"]
    worker.stop_ffn_server_loop()
    assert worker._ffn_thread is None
