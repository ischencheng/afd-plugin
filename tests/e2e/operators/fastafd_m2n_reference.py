# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Real-checkpoint check of the FastAFD M2N MegaMoE kernels on SM100.

Run with one process per GPU; the last ``--num-ffn-ranks`` ranks are FFN
(expert) ranks and the rest are Attention ranks, matching the connector's
Attention-first union layout:

    torchrun --nproc-per-node 2 -m tests.e2e.operators.fastafd_m2n_reference \
        --model /path/to/Qwen3.6-35B-A3B-FP8 --output /logs/fastafd-m2n

FFN ranks requantize their checkpoint experts with the plugin helpers and run
one persistent expert kernel per step over all selected layers and lanes.
Attention ranks route fixed inputs with the checkpoint gate (softmax top-k,
renormalized) and run the dispatch/combine kernel per layer and lane. Each
output is compared with two fp32 references:

- matched: the kernel's numerics (per-32 UE8M0 FP8 activations, requantized
  weights, FP8 intermediate after SwiGLU and the top-k weight);
- model: block-FP8 checkpoint weights with unquantized activations.

The matched bound checks the kernel; the model figure reports the error the
M2N quantization adds relative to the checkpoint math.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open

from afd_plugin.compat.gpu.fastafd import (
    FASTAFD_CHECKPOINT_BLOCK,
    FASTAFD_SCALE_GROUP,
    ceil_to_ue8m0,
    fastafd_ops,
    make_weight_descriptors,
    requant_fp8_block_to_per32,
    transform_weights_for_m2n,
)
from afd_plugin.connectors.gpu.fastafd import M2NLaneBuffer, M2NShape

TOKEN_COUNTS = (1, 7, 64, 256)
SEED = 1234
# Relative L2 bound against the matched reference: both sides quantize and
# round at the same points, so the gap is accumulation order, fast-math SiLU
# and FP8 values that land on a rounding boundary.
MATCHED_REL_L2 = 0.01
FP8_MAX = 448.0
# The dispatch kernel floors each 32-element input amax at 1e-4; the expert
# kernel uses the raw amax of the intermediate.
INPUT_AMAX_FLOOR = 1e-4
INTERMEDIATE_AMAX_FLOOR = 1e-30
ATTENTION_SMS = 24


def text_config(model: Path) -> dict:
    config = json.loads((model / "config.json").read_text())
    return config.get("text_config", config)


class Checkpoint:
    def __init__(self, model: Path) -> None:
        self.model = model
        self.index = json.loads((model / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

    def tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        by_file: dict[str, list[str]] = {}
        for name in names:
            by_file.setdefault(self.index[name], []).append(name)
        out = {}
        for file_name, file_names in by_file.items():
            with safe_open(str(self.model / file_name), framework="pt") as file:
                for name in file_names:
                    out[name] = file.get_tensor(name)
        return out


def layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}.mlp"


def expert_weights(
    checkpoint: Checkpoint, layer: int, experts: range
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stacked FP8 ``gate_up [E, 2I, H]``, ``down [E, H, I]`` and block scales."""
    prefix = layer_prefix(layer)
    names = [
        f"{prefix}.experts.{expert}.{proj}.{kind}"
        for expert in experts
        for proj in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale_inv")
    ]
    tensors = checkpoint.tensors(names)

    def stack(proj: str, kind: str) -> torch.Tensor:
        return torch.stack(
            [tensors[f"{prefix}.experts.{e}.{proj}.{kind}"] for e in experts]
        ).cuda()

    gate_up = torch.cat(
        [stack("gate_proj", "weight"), stack("up_proj", "weight")], dim=1
    )
    gate_up_scales = torch.cat(
        [stack("gate_proj", "weight_scale_inv"), stack("up_proj", "weight_scale_inv")],
        dim=1,
    )
    return (
        gate_up,
        gate_up_scales,
        stack("down_proj", "weight"),
        stack("down_proj", "weight_scale_inv"),
    )


def block_dequant(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    expanded = torch.repeat_interleave(scales, FASTAFD_CHECKPOINT_BLOCK, dim=-2)
    expanded = torch.repeat_interleave(expanded, FASTAFD_CHECKPOINT_BLOCK, dim=-1)
    return weight.float() * expanded[..., : weight.shape[-2], : weight.shape[-1]]


def requant_dequant(weight: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """fp32 values of the per-32 UE8M0 weights the kernel consumes."""
    rounded = ceil_to_ue8m0(scales.float())
    ratio = torch.where(
        rounded > 0, scales.float() / rounded, torch.zeros_like(rounded)
    )
    requantized = (block_dequant(weight, ratio)).to(torch.float8_e4m3fn)
    return block_dequant(requantized, rounded)


def fp8_per32(x: torch.Tensor, amax_floor: float) -> torch.Tensor:
    """fp32 values after per-32 UE8M0 FP8 quantization, as the kernels do it."""
    groups = x.float().reshape(*x.shape[:-1], -1, FASTAFD_SCALE_GROUP)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(amax_floor)
    scales = ceil_to_ue8m0(amax / FP8_MAX)
    quantized = (groups / scales).to(torch.float8_e4m3fn).float() * scales
    return quantized.reshape(x.shape)


def moe_reference(
    hidden: torch.Tensor,
    ids: torch.Tensor,
    weights: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    *,
    matched: bool,
) -> torch.Tensor:
    """MoE output; ``matched`` follows the kernels' quantization and rounding.

    The matched path quantizes inputs per 32 with an amax floor of 1e-4,
    rounds gate and up to bfloat16 before SwiGLU, applies the top-k weight
    before quantizing the intermediate, and rounds every expert output to
    bfloat16 before the fp32 top-k sum.
    """
    x = fp8_per32(hidden, INPUT_AMAX_FLOOR) if matched else hidden.float()
    intermediate = gate_up.shape[1] // 2
    out = torch.zeros_like(x)
    for expert in ids.unique().tolist():
        rows, slots = (ids == expert).nonzero(as_tuple=True)
        projected = x[rows] @ gate_up[expert].T
        if matched:
            projected = projected.to(torch.bfloat16).float()
        activated = torch.nn.functional.silu(projected[:, :intermediate])
        activated = activated * projected[:, intermediate:]
        activated = activated * weights[rows, slots, None]
        if matched:
            activated = fp8_per32(activated, INTERMEDIATE_AMAX_FLOOR)
        expert_out = activated @ down[expert].T
        if matched:
            expert_out = expert_out.to(torch.bfloat16).float()
        out.index_add_(0, rows, expert_out)
    return out


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = actual.float() - expected.float()
    return (error.norm() / expected.float().norm().clamp_min(1e-12)).item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--lanes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--num-ffn-ranks", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    hf = text_config(args.model)
    num_attention = world - args.num_ffn_ranks
    shape = M2NShape(
        num_attention_ranks=num_attention,
        num_ffn_ranks=args.num_ffn_ranks,
        num_experts=hf["num_experts"],
        top_k=hf["num_experts_per_tok"],
        hidden_size=hf["hidden_size"],
        intermediate_size=hf["moe_intermediate_size"],
        max_tokens_per_rank=args.max_tokens,
    )
    group = dist.group.WORLD
    buffers = [M2NLaneBuffer(shape, group) for _ in range(max(args.lanes))]
    ops = fastafd_ops()
    checkpoint = Checkpoint(args.model)
    steps = [(lanes, count) for lanes in args.lanes for count in TOKEN_COUNTS]
    layers = args.layers

    if rank >= num_attention:
        ffn_rank = rank - num_attention
        local = shape.num_experts_per_ffn_rank
        experts = range(ffn_rank * local, (ffn_rank + 1) * local)
        kept, blobs, l1_ptrs = [], [], []
        for layer in layers:
            gate_up, gate_up_scales, down, down_scales = expert_weights(
                checkpoint, layer, experts
            )
            l1, l2 = transform_weights_for_m2n(
                requant_fp8_block_to_per32(gate_up, gate_up_scales),
                requant_fp8_block_to_per32(down, down_scales),
            )
            kept.append((l1, l2))
            blobs.append(
                make_weight_descriptors(
                    l1,
                    l2,
                    num_experts_per_rank=local,
                    hidden_size=shape.hidden_size,
                    intermediate_size=shape.intermediate_size,
                ).cuda()
            )
            l1_ptrs.append(l1[0].data_ptr())
        descriptors = torch.stack(blobs).contiguous()
        pointers = torch.tensor(l1_ptrs, dtype=torch.int64, device="cuda")
        dist.barrier(group)
        for lanes, count in steps:
            lane_buffers = [
                buffers[lane] if lane < lanes else None for lane in range(4)
            ]
            ptrs = [
                (lane_buffer or buffers[0]).peer_ptrs for lane_buffer in lane_buffers
            ]
            ops.mega_moe_m2n_eg(
                descriptors,
                pointers,
                len(layers),
                local,
                shape.hidden_size,
                shape.intermediate_size,
                kept[0][0][0].numel(),
                True,
                None,
                buffers[0].buffer,
                *(
                    lane_buffer.buffer if lane_buffer else None
                    for lane_buffer in lane_buffers[1:]
                ),
                *ptrs,
                rank,
                num_attention,
                args.num_ffn_ranks,
                buffers[0].max_tokens_per_rank,
                shape.num_experts,
                shape.top_k,
                count,
                None,
                True,
                0,
                0,
                None,
            )
            torch.cuda.synchronize()
        dist.barrier(group)
        dist.destroy_process_group()
        return

    gates = {
        layer: checkpoint.tensors([f"{layer_prefix(layer)}.gate.weight"])[
            f"{layer_prefix(layer)}.gate.weight"
        ].cuda()
        for layer in layers
    }
    reference_weights = {}
    for layer in layers:
        gate_up, gate_up_scales, down, down_scales = expert_weights(
            checkpoint, layer, range(shape.num_experts)
        )
        reference_weights[layer] = {
            "matched": (
                requant_dequant(gate_up, gate_up_scales),
                requant_dequant(down, down_scales),
            ),
            "model": (
                block_dequant(gate_up, gate_up_scales),
                block_dequant(down, down_scales),
            ),
        }
    norm_topk = bool(hf.get("norm_topk_prob", True))
    dist.barrier(group)
    reports = []
    for lanes, count in steps:
        inputs = []
        for lane in range(lanes):
            generator = torch.Generator(device="cuda").manual_seed(
                SEED + 101 * rank + 17 * count + lane
            )
            raw = torch.randn(
                count, shape.hidden_size, device="cuda", generator=generator
            )
            inputs.append(
                (raw / raw.pow(2).mean(-1, keepdim=True).sqrt()).to(torch.bfloat16)
            )
        outputs = {}
        routes = {}
        for layer in layers:
            for lane, hidden in enumerate(inputs):
                logits = torch.nn.functional.linear(hidden, gates[layer])
                probs = logits.float().softmax(dim=-1)
                weights, ids = probs.topk(shape.top_k, dim=-1)
                if norm_topk:
                    weights = weights / weights.sum(dim=-1, keepdim=True)
                ids = ids.to(torch.int32).contiguous()
                weights = weights.contiguous()
                y = torch.empty_like(hidden)
                ops.mega_moe_m2n_ag(
                    y,
                    hidden,
                    ids,
                    weights,
                    None,
                    norm_topk,
                    buffers[lane].buffer,
                    buffers[lane].peer_ptrs,
                    rank,
                    num_attention,
                    args.num_ffn_ranks,
                    buffers[lane].max_tokens_per_rank,
                    shape.num_experts,
                    shape.top_k,
                    shape.intermediate_size,
                    ATTENTION_SMS,
                    0,
                    False,
                    False,
                    shape.top_k,
                    shape.num_experts,
                    0.0,
                    None,
                )
                outputs[(layer, lane)] = y
                routes[(layer, lane)] = (ids, weights)
        torch.cuda.synchronize()
        for (layer, lane), y in outputs.items():
            ids, weights = routes[(layer, lane)]
            report = {
                "rank": rank,
                "lanes": lanes,
                "tokens": count,
                "layer": layer,
                "lane": lane,
                "finite": bool(torch.isfinite(y).all().item()),
            }
            for kind in ("matched", "model"):
                gate_up, down = reference_weights[layer][kind]
                reference = moe_reference(
                    inputs[lane],
                    ids.long(),
                    weights,
                    gate_up,
                    down,
                    matched=kind == "matched",
                )
                report[f"{kind}_rel_l2"] = rel_l2(y, reference)
                report[f"{kind}_max_abs"] = (y.float() - reference).abs().max().item()
            reports.append(report)
            print(json.dumps(report), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"report-rank{rank}.json").write_text(json.dumps(reports, indent=2))
    dist.barrier(group)
    dist.destroy_process_group()
    for report in reports:
        assert report["finite"], report
        assert report["matched_rel_l2"] <= MATCHED_REL_L2, report


if __name__ == "__main__":
    main()
