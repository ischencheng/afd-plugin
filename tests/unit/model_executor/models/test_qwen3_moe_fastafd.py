# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Qwen3 MoE on FastAFDAFDConnector: Attention routes and drives dispatch/combine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.connectors.gpu.fastafd import FastAFDAFDConnector  # noqa: E402
from afd_plugin.model_executor.models import fastafd_moe  # noqa: E402
from afd_plugin.model_executor.models import qwen3_moe as adapter  # noqa: E402

FASTAFD = "FastAFDAFDConnector"


class _FakeGate(nn.Module):
    def __init__(self, input_size, output_size, *, bias, quant_config, prefix):
        super().__init__()
        self.prefix = prefix
        self.quant_config = quant_config
        self.weight = nn.Parameter(torch.zeros(output_size, input_size))

    def forward(self, hidden_states):
        return hidden_states[:, :4], None


def _vllm_config(*, mlp_only_layers=None, shared_expert_intermediate_size=None):
    config = SimpleNamespace(
        attention_bias=False,
        decoder_sparse_step=1,
        head_dim=8,
        hidden_act="silu",
        hidden_size=16,
        intermediate_size=32,
        max_position_embeddings=4096,
        mlp_only_layers=mlp_only_layers or [],
        norm_topk_prob=True,
        num_attention_heads=2,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=4,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 1_000_000.0},
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        tie_word_embeddings=False,
        vocab_size=128,
    )
    return SimpleNamespace(
        cache_config=None,
        model_config=SimpleNamespace(hf_text_config=config),
        quant_config="fp8-quant-config",
    )


@pytest.fixture
def attention_layer_env(monkeypatch):
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(
            role="attention",
            connector=FASTAFD,
            compute_gate_on_attention=True,
        ),
    )
    monkeypatch.setattr(adapter.native, "ReplicatedLinear", _FakeGate)
    monkeypatch.setattr(
        adapter.native, "Qwen3MoeAttention", lambda **_kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        adapter.native, "RMSNorm", lambda *_args, **_kwargs: nn.Identity()
    )


def test_attention_layer_builds_the_router_with_the_checkpoint_quant_config(
    attention_layer_env,
):
    layer = adapter.AFDQwen3MoeDecoderLayer(_vllm_config(), "model.layers.2")

    assert isinstance(layer.mlp, adapter.AFDQwen3MoeAttentionRoutedMoE)
    assert layer.mlp.gate.prefix == "model.layers.2.mlp.gate"
    # The native router uses the checkpoint quant config, which skips the gate
    # of FP8 Qwen3 checkpoints (modules_to_not_convert).
    assert layer.mlp.gate.quant_config == "fp8-quant-config"
    assert (layer.mlp.top_k, layer.mlp.renormalize) == (2, True)
    assert "mlp.gate.weight" in dict(layer.named_parameters())


def test_attention_layer_rejects_dense_layers(attention_layer_env):
    with pytest.raises(RuntimeError, match="MoE block in every layer"):
        adapter.AFDQwen3MoeDecoderLayer(
            _vllm_config(mlp_only_layers=[1]),
            "model.layers.1",
        )


def test_attention_layer_rejects_shared_experts(attention_layer_env):
    with pytest.raises(RuntimeError, match="shared experts"):
        adapter.AFDQwen3MoeDecoderLayer(
            _vllm_config(shared_expert_intermediate_size=64),
            "model.layers.0",
        )


def _routed_moe(monkeypatch, connector, events):
    moe = object.__new__(adapter.AFDQwen3MoeAttentionRoutedMoE)
    nn.Module.__init__(moe)
    moe.gate = lambda hidden_states: (hidden_states[:, :4], None)
    moe.top_k = 2
    moe.renormalize = True
    monkeypatch.setattr(
        adapter,
        "fused_topk",
        lambda hidden_states, logits, top_k, renormalize: (
            torch.full((3, 2), 0.5),
            torch.zeros(3, 2, dtype=torch.int32),
            None,
        ),
    )
    monkeypatch.setattr(
        fastafd_moe,
        "get_afd_metadata_from_forward_context",
        lambda: SimpleNamespace(connector=connector, stage_idx=0),
    )
    monkeypatch.setattr(fastafd_moe, "get_forward_context", lambda: SimpleNamespace())
    # The custom ops are registered for the platform dispatch key, which is
    # CUDA on GPU hosts, so call their implementations directly on CPU tensors.
    monkeypatch.setattr(
        adapter,
        "torch",
        SimpleNamespace(
            ops=SimpleNamespace(
                vllm=SimpleNamespace(
                    afd_fastafd_dispatch=fastafd_moe._fastafd_dispatch,
                    afd_fastafd_combine=fastafd_moe._fastafd_combine,
                ),
            ),
        ),
    )
    return moe


def test_routed_moe_dispatches_then_combines_on_the_step_lane(monkeypatch):
    events: list[tuple[str, int]] = []

    def dispatch(hidden_states, ids, weights, lane):
        events.append(("dispatch", lane))

    def combine(hidden_states, ids, weights, lane):
        events.append(("combine", lane))
        return hidden_states * 2

    connector = object.__new__(FastAFDAFDConnector)
    connector.dispatch = dispatch  # type: ignore[method-assign]
    connector.combine = combine  # type: ignore[method-assign]
    moe = _routed_moe(monkeypatch, connector, events)
    hidden_states = torch.arange(24, dtype=torch.float32).reshape(3, 8)

    output = moe(hidden_states)

    assert events == [("dispatch", 0), ("combine", 0)]
    assert torch.equal(output, hidden_states * 2)


def test_routed_moe_requires_the_fastafd_connector(monkeypatch):
    moe = _routed_moe(monkeypatch, object(), [])

    with pytest.raises(RuntimeError, match="requires FastAFDAFDConnector"):
        moe(torch.zeros(3, 8))
