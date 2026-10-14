# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Graph DR slot/STOP tests using the original DS/CS/CR communication ABI.

Run with torchrun --nproc-per-node=2 (or 3 with --zero-rank).
Every fresh process first performs an ordinary paired warmup. No standalone
initialization operator or controlled CS is used. The runtime fixture covers
capture and the complete DR/FFN/CS graph; this fixture isolates native transport.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from afd_plugin.compat.npu.ops import ensure_cam_ffn_graph_available
from tests.npu.async_cam_common import HarnessConfig, NativeHarness

STOP_TIMEOUT_SECONDS = 30
IDLE_PROOF_SECONDS = 0.5
INVALID_TEST_LAYER = 3


def wait_complete(event):
    deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
    while not event.query():
        if time.monotonic() >= deadline:
            raise TimeoutError("Native completion exceeded 30 seconds")
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--invalid-layer", action="store_true")
    parser.add_argument("--zero-rank", action="store_true")
    args = parser.parse_args()
    config = HarnessConfig(
        attn_ranks=1,
        moe_ranks=2 if args.zero_rank else 1,
        tp_size=1,
        lengths=(7,),
        max_seq_len=16,
        dynamic_quant=1,
        routing="sparse" if args.zero_rank else "balanced",
    )
    ensure_cam_ffn_graph_available()
    with NativeHarness(config) as h, torch.inference_mode():
        warmup = h.run_round(layer_id=2)
        assert not warmup["failures"], warmup
        report = {"rank": h.rank, "paired_warmup": warmup, "work": []}
        ops = torch.ops.afd_ascend
        is_ffn = h.rank >= config.attn_ranks
        if is_ffn:
            stop = torch.zeros(2, dtype=torch.int64, device=h.device)
            layers = torch.tensor([2, 5], dtype=torch.int64, device=h.device)
            slots = []

            def receive(*values):
                outputs = ops.afd_async_dispatch_recv_controlled(*values, stop, layers)
                slots.append(outputs[4])
                return outputs[:4]

            h.operations["dispatch_recv"] = receive
        for layer_id in (2, 5, 2):
            result = h.run_round(layer_id=layer_id)
            assert not result["failures"], result
            if is_ffn:
                assert all(int(slot.item()) == (layer_id == 5) for slot in slots)
                slots.clear()
                if args.zero_rank and h.rank == 2:
                    assert result["rows"] == 0 and result["chunks"] == 1
            report["work"].append(result)

        common = (
            config.hidden_size,
            config.top_k,
            config.moe_ranks,
            config.attn_ranks,
            config.experts_per_rank,
            h.rank,
            config.world_size,
        )
        if is_ffn:
            service = torch.npu.Stream(device=h.device)
            signal_stream = torch.npu.Stream(device=h.device)
            output = torch.zeros(
                (config.capacity, config.hidden_size), dtype=h.dtype, device=h.device
            )
            stop_host = torch.ones(1, dtype=torch.int64, pin_memory=True)
            host_slot = torch.empty(1, dtype=torch.int64, pin_memory=True)
            started, done, signal_done = (torch.npu.Event() for _ in range(3))
            service.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(service):
                started.record(service)
                for _ in range(2):
                    received = ops.afd_async_dispatch_recv_controlled(
                        h.anchor,
                        h.comm_args,
                        0,
                        config.max_seq_len,
                        *common,
                        config.tp_size,
                        config.dynamic_quant,
                        h.group_name,
                        stop,
                        layers,
                    )
                    placeholder = ops.afd_async_combine_send(
                        output,
                        h.comm_args,
                        received[2],
                        0,
                        config.max_seq_len,
                        *common,
                        config.tp_size,
                        h.group_name,
                    )
                host_slot.copy_(received[4], non_blocking=True)
                done.record(service)
            wait_complete(started)
            time.sleep(IDLE_PROOF_SECONDS)
            assert not done.query(), "Idle DR must wait until work or STOP"
        # Every A has completed CR for every normal round. This Gloo barrier
        # supplies the fixture's explicit drain proof before any STOP write.
        h.dist.barrier(group=h.control_group)
        if args.invalid_layer:
            if not is_ffn:
                inputs, ids, _ = (
                    tensor.to(h.device)
                    for tensor in h._input(h.rank, INVALID_TEST_LAYER)
                )
                dispatched = ops.afd_async_dispatch_send(
                    inputs,
                    ids,
                    h.comm_args,
                    0,
                    config.max_seq_len,
                    config.lengths[h.rank],
                    *common,
                    INVALID_TEST_LAYER,
                    config.tp_size,
                    config.dynamic_quant,
                    h.group_name,
                )
                ds_done = torch.npu.Event()
                ds_done.record()
                wait_complete(ds_done)
                assert dispatched is not None
            # Deliberately no CR for invalid work; this is error-path evidence,
            # not a normal shutdown or retry/recovery of the failed service.
            report["normal_stop"] = False
        elif is_ffn:
            with torch.npu.stream(signal_stream):
                stop[:1].copy_(stop_host, non_blocking=True)
                signal_done.record(signal_stream)
            wait_complete(signal_done)
            report["stop_copy_fenced"] = True
        if is_ffn:
            wait_complete(done)
            slot = int(host_slot[0])
            assert slot == (-2 if args.invalid_layer else -1), slot
            assert int(received[2][2].item()) < 0
            assert int(received[3].sum().item()) == 0
            assert placeholder is not None
            report["terminal_slot"] = slot
            report["queued_receives"] = 2
            if args.invalid_layer:
                assert stop.cpu().tolist() == [0, -2]
                # Host STOP must not hide a prior error. Make the stale header's
                # layer valid as well: the next receive must keep the error
                # rather than consume that header or interpret it as STOP.
                for request_stop in (False, True):
                    with torch.npu.stream(signal_stream):
                        layers.fill_(INVALID_TEST_LAYER)
                        if request_stop:
                            stop[:1].copy_(stop_host, non_blocking=True)
                        signal_done.record(signal_stream)
                    wait_complete(signal_done)
                    with torch.npu.stream(service):
                        received = ops.afd_async_dispatch_recv_controlled(
                            h.anchor,
                            h.comm_args,
                            0,
                            config.max_seq_len,
                            *common,
                            config.tp_size,
                            config.dynamic_quant,
                            h.group_name,
                            stop,
                            layers,
                        )
                        host_slot.copy_(received[4], non_blocking=True)
                        done.record(service)
                    wait_complete(done)
                    assert int(host_slot[0]) == -2
                    assert stop.cpu().tolist() == [int(request_stop), -2]
                    assert int(received[2][2]) == -2 and int(received[3].sum()) == 0
                report["first_error_survives_stop_and_valid_header"] = True
        # All submitted device work is fenced before transport and tensors die.
        h.dist.barrier(group=h.control_group)
        reports = h.gather_reports(report)
        if h.rank == 0:
            print(json.dumps({"result": "PASS", "reports": reports}), flush=True)


if __name__ == "__main__":
    main()
