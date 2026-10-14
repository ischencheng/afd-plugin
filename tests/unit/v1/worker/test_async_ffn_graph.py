# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph ownership contracts; fake graphs do not qualify NPU capture."""

import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")
pytest.importorskip("torch_npu")
import torch  # noqa: E402

from afd_plugin.v1.worker.npu.async_ffn_model_runner import (  # noqa: E402
    AsyncNPUFFNModelRunner,
)


@pytest.fixture
def session_harness(monkeypatch):
    calls = []
    slot = torch.tensor([0])
    connector = SimpleNamespace(next_slot=0, failure=None)

    class FakeGraph:
        fail_launch = False

        def replay(self):
            calls.append("replay")
            slot.fill_(connector.next_slot)
            if self.fail_launch:
                raise RuntimeError("partial launch")

        def reset(self):
            calls.append("reset")

    class FakeEvent:
        ready = True

        def query(self):
            return self.ready

        def record(self, stream):
            calls.append("record")

        def synchronize(self):
            pytest.fail("business completion uses one batch device fence")

    @contextmanager
    def capture(graph, *, pool, stream):
        calls.append("capture")
        yield

    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(
            NPUGraph=FakeGraph,
            Event=FakeEvent,
            graph=capture,
            graph_pool_handle=lambda: (1, 0),
            stream=lambda stream: nullcontext(),
            synchronize=lambda: calls.append("wait"),
        ),
        raising=False,
    )
    original_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, pin_memory=False, **kwargs: original_empty(*args, **kwargs),
    )
    state = SimpleNamespace(
        layer_slot=slot, group_list=torch.zeros(2), dynamic_scales=torch.ones(4)
    )
    payload = SimpleNamespace(
        hidden_states=torch.ones(4), context=SimpleNamespace(states=state)
    )

    def receive():
        calls.append("receive")
        return payload

    def execute(hidden, scales, counts, layer_slot, retained):
        calls.append("compute")
        assert layer_slot is slot
        temporary = hidden.clone()
        retained.append(temporary)
        return temporary * 2

    def send(output, context):
        assert context is payload.context
        calls.append("send")

    def complete(observed):
        calls.append("complete")
        assert observed == connector.next_slot
        if connector.failure:
            raise connector.failure
        return observed >= 0

    connector.recv_attn_output = receive
    connector.send_ffn_output = send
    connector.finish_graph = complete
    connector.poll_graph_stop = lambda: calls.append("poll-stop")
    session = AsyncNPUFFNModelRunner.__new__(AsyncNPUFFNModelRunner)
    session.model = torch.nn.Linear(1, 1)
    session._layered_executor = SimpleNamespace(execute_graph=execute)
    session.connector = connector
    session._graph = None
    session._graph_capture_count = session._graph_replay_count = 0
    session._graph_pending = session._graph_stopped = session._graph_closed = False
    session._graph_failure = None
    session._initialize_graph(SimpleNamespace(synchronize=lambda: calls.append("wait")))
    return session, connector, calls


def test_replay_retains_transaction_until_terminal_completion(session_harness):
    session, connector, calls = session_harness
    session._capture_graph()
    temporary = weakref.ref(session._graph_tensors[0])
    for slot in (0, 1, 0):
        connector.next_slot = slot
        session._submit_graph()
        assert session._complete_graph()
    assert calls.count("receive") == calls.count("compute") == calls.count("send") == 1
    assert calls.count("replay") == 3 and temporary() is not None
    with pytest.raises(RuntimeError, match="terminal completion"):
        session._close_graph()
    connector.next_slot = -1
    session._submit_graph()
    assert not session._complete_graph()
    session._close_graph()
    session._close_graph()
    assert calls.count("reset") == 1 and temporary() is None


def test_next_batch_cannot_start_while_pending(session_harness):
    session, _, calls = session_harness
    session._capture_graph()
    session._submit_graph()
    with pytest.raises(RuntimeError, match="not ready"):
        session._submit_graph()
    with pytest.raises(RuntimeError, match="terminal completion"):
        session._close_graph()
    assert calls.count("replay") == 1 and "reset" not in calls


@pytest.mark.parametrize("count", [1, 43])
def test_batch_waits_once_without_polling_store(session_harness, monkeypatch, count):
    session, connector, calls = session_harness

    def unexpected(*args):
        pytest.fail("business completion must not poll or sleep")

    connector.poll_graph_stop = unexpected
    monkeypatch.setattr(
        "afd_plugin.v1.worker.npu.async_ffn_model_runner.time.sleep", unexpected
    )
    session._capture_graph()
    copies = []
    session._graph_host_slot = SimpleNamespace(
        copy_=lambda slot, **kwargs: copies.append(slot)
    )
    session._submit_graph(count)
    assert copies == [session._graph_slot]
    session._graph_host_slot = [connector.next_slot]
    session._graph_done.query = unexpected
    assert session._complete_graph()
    assert calls.count("replay") == session._graph_replay_count == count
    assert calls.count("record") == calls.count("wait") == 1
    assert calls[-2:] == ["wait", "complete"]


@pytest.mark.parametrize("failure", ["submit", "slot", "timeout", "drain"])
def test_failure_retains_resources_without_retry(session_harness, failure):
    session, connector, calls = session_harness
    session._capture_graph()
    if failure == "submit":
        session._graph.fail_launch = True
        operation = session._submit_graph
    else:
        session._submit_graph()
        if failure == "slot":
            connector.failure = RuntimeError("invalid device slot")
        else:
            session._graph_done.ready = False
            if failure == "drain":

                def fail():
                    raise RuntimeError("drain failed")

                connector.poll_graph_stop = fail

        def operation():
            return session._complete_graph(timeout=0)

    with pytest.raises((RuntimeError, TimeoutError)):
        operation()
    for operation in (
        session._submit_graph,
        session._complete_graph,
        session._close_graph,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
    assert session._graph_payload is not None and session._graph_output is not None
    assert "reset" not in calls


def test_partial_capture_never_retries_or_releases(session_harness):
    session, connector, calls = session_harness

    def fail_send(*args):
        raise RuntimeError("capture failed after receive")

    connector.send_ffn_output = fail_send
    with pytest.raises(RuntimeError, match="capture failed"):
        session._capture_graph()
    for operation in (
        session._capture_graph,
        session._submit_graph,
        session._close_graph,
    ):
        with pytest.raises(RuntimeError, match="failed"):
            operation()
    assert session._graph_payload is not None and session._graph_output is not None
    assert "replay" not in calls and "reset" not in calls
