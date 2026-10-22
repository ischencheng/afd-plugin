# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN runner that executes each AFD step as one FastAFD expert-kernel launch.

The runner follows the megakernel FFN structure proposed for #391: the expert
weights (``FastAFDExpertWeights``) are loaded FFN-only and converted to the
kernel layout, and every step plan from the control plane becomes a single
persistent kernel launch over all MoE layers and microbatch lanes. The kernel
receives tokens, computes the experts and writes results back by itself; the
host never splits it into per-layer receive/compute/send.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from safetensors import safe_open
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.utils.mem_utils import DeviceMemoryProfiler

from afd_plugin.compat.gpu.fastafd import (
    fastafd_ops,
    make_weight_descriptors,
    requant_fp8_block_to_per32,
    transform_weights_for_m2n,
)
from afd_plugin.compat.profiler import step_afd_gpu_profiler
from afd_plugin.connectors.gpu.fastafd import FastAFDAFDConnector, M2NShape
from afd_plugin.model_executor.models.qwen3_5 import fastafd_expert_tensor_name
from afd_plugin.v1.worker.ffn_model_runner import GPUFFNModelRunner

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import DPMetadata

    from afd_plugin.connectors import AFDDPMetadata

# The expert kernel serves at most this many microbatch lanes per launch.
_MAX_KERNEL_LANES = 4
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _checkpoint_dir(vllm_config: VllmConfig) -> Path:
    model_config = vllm_config.model_config
    if os.path.isdir(model_config.model):
        return Path(model_config.model)
    return Path(
        download_weights_from_hf(
            model_config.model,
            vllm_config.load_config.download_dir,
            allow_patterns=["*.safetensors", "*.json"],
            revision=model_config.revision,
        )
    )


class FastAFDExpertWeights:
    """This FFN rank's routed experts in the FastAFD kernel layout.

    Loads only the local experts' FP8 tensors from the checkpoint, requantizes
    them to per-32 UE8M0, and keeps the converted tensors alive for the TMA
    descriptors that point at them.
    """

    def __init__(self, vllm_config: VllmConfig, shape: M2NShape, ffn_rank: int) -> None:
        config = vllm_config.model_config.hf_text_config
        self.num_layers = int(config.num_hidden_layers)
        local = shape.num_experts_per_ffn_rank
        experts = range(ffn_rank * local, (ffn_rank + 1) * local)
        checkpoint = _checkpoint_dir(vllm_config)
        weight_map = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]
        self.layers: list[
            tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
        ] = []
        descriptors = []
        for layer in range(self.num_layers):
            l1, l2 = _load_layer_experts(checkpoint, weight_map, layer, experts)
            self.layers.append((l1, l2))
            descriptors.append(
                make_weight_descriptors(
                    l1,
                    l2,
                    num_experts_per_rank=local,
                    hidden_size=shape.hidden_size,
                    intermediate_size=shape.intermediate_size,
                ).cuda()
            )
        self.descriptors = torch.stack(descriptors).contiguous()
        self.l1_pointers = torch.tensor(
            [l1[0].data_ptr() for l1, _ in self.layers],
            dtype=torch.int64,
            device="cuda",
        )
        self.l1_nbytes = self.layers[0][0][0].numel()


def _load_layer_experts(
    checkpoint: Path,
    weight_map: dict[str, str],
    layer: int,
    experts: range,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """One layer's local experts as kernel-layout ``(L1, L2)`` weight pairs."""
    tensors = _read_tensors(
        checkpoint,
        weight_map,
        [
            fastafd_expert_tensor_name(layer, expert, projection, kind)
            for expert in experts
            for projection in _PROJECTIONS
            for kind in ("weight", "weight_scale_inv")
        ],
    )

    def stack(projection: str, kind: str) -> torch.Tensor:
        return torch.stack(
            [
                tensors[fastafd_expert_tensor_name(layer, expert, projection, kind)]
                for expert in experts
            ]
        ).cuda()

    gate_up = torch.cat(
        [stack("gate_proj", "weight"), stack("up_proj", "weight")], dim=1
    )
    gate_up_scales = torch.cat(
        [stack("gate_proj", "weight_scale_inv"), stack("up_proj", "weight_scale_inv")],
        dim=1,
    )
    down = stack("down_proj", "weight")
    down_scales = stack("down_proj", "weight_scale_inv")
    return transform_weights_for_m2n(
        requant_fp8_block_to_per32(gate_up, gate_up_scales),
        requant_fp8_block_to_per32(down, down_scales),
    )


def _read_tensors(
    checkpoint: Path,
    weight_map: dict[str, str],
    names: list[str],
) -> dict[str, torch.Tensor]:
    by_file: dict[str, list[str]] = {}
    for name in names:
        by_file.setdefault(weight_map[name], []).append(name)
    tensors = {}
    for file_name, file_names in by_file.items():
        with safe_open(str(checkpoint / file_name), framework="pt") as file:
            for name in file_names:
                tensors[name] = file.get_tensor(name)
    return tensors


class FastAFDFFNModelRunner(GPUFFNModelRunner):
    """Connector-driven FFN runner for ``FastAFDAFDConnector``."""

    connector: FastAFDAFDConnector

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        super().__init__(vllm_config, device)
        self.expert_weights: FastAFDExpertWeights | None = None

    def load_model(self, *, load_dummy_weights: bool = False, **kwargs: Any) -> None:
        """Load this rank's routed experts straight into the kernel layout."""
        if load_dummy_weights:
            raise ValueError("FastAFDAFDConnector needs checkpoint expert weights")
        fastafd_ops()
        with DeviceMemoryProfiler() as profiler:
            self.expert_weights = FastAFDExpertWeights(
                self.vllm_config,
                self.connector.shape,
                self.connector.role_rank,
            )
        self.model_memory_usage = profiler.consumed_memory

    def execute_model(
        self,
        scheduler_output: Any = None,
        intermediate_tensors: Any = None,
        *,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_graph_capturing: bool = False,
        is_warmup: bool = False,
        is_graph_replaying: bool = False,
    ) -> None:
        step_afd_gpu_profiler(self.prof)
        if dp_metadata_list is None:
            raise RuntimeError("FastAFDFFNModelRunner requires dp_metadata_list")
        # Attention records its kernels into a graph without running them, so
        # a capture step has no expert work to serve.
        if not is_graph_capturing:
            self.submit_step(dp_metadata_list)

    def capture_model(
        self,
        dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata] | None = None,
        is_warmup: bool = False,
        is_attn_graph_capturing: bool = True,
    ) -> int:
        if dp_metadata_list is None:
            raise RuntimeError("FastAFDFFNModelRunner requires dp_metadata_list")
        if is_warmup:
            self.submit_step(dp_metadata_list)
        return 0

    def submit_step(
        self, dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata]
    ) -> None:
        """Launch the expert kernel for one step: every MoE layer and lane."""
        weights = self.expert_weights
        assert weights is not None
        connector = self.connector
        shape = connector.shape
        num_lanes = len(dp_metadata_list)
        lanes = connector.lanes[:num_lanes]
        extra_lanes = [
            lanes[i] if i < num_lanes else None for i in range(1, _MAX_KERNEL_LANES)
        ]
        expected_tokens = max(
            max(int(count) for count in metadata.num_tokens_across_dp_cpu)
            for metadata in dp_metadata_list.values()
        )
        fastafd_ops().mega_moe_m2n_eg(
            weights.descriptors,
            weights.l1_pointers,
            weights.num_layers,
            shape.num_experts_per_ffn_rank,
            shape.hidden_size,
            shape.intermediate_size,
            weights.l1_nbytes,
            True,
            None,
            lanes[0].buffer,
            *(lane.buffer if lane is not None else None for lane in extra_lanes),
            lanes[0].peer_ptrs,
            *((lane or lanes[0]).peer_ptrs for lane in extra_lanes),
            connector.union_rank,
            shape.num_attention_ranks,
            shape.num_ffn_ranks,
            lanes[0].max_tokens_per_rank,
            shape.num_experts,
            shape.top_k,
            max(1, expected_tokens),
            None,
            True,
            0,
            0,
            None,
        )

    def shutdown(self) -> None:
        self.expert_weights = None
        super().shutdown()


__all__ = ["FastAFDExpertWeights", "FastAFDFFNModelRunner"]
