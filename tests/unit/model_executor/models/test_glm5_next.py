# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torch import Tensor

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402

from afd_plugin.config import AFDConfig  # noqa: E402
from afd_plugin.model_executor.models import glm5_next as adapter  # noqa: E402
from tests.unit.fakes.model_layers import stage_type as _stage_type  # noqa: E402
from tests.unit.fakes.weights import OneShotWeights as _OneShotWeights  # noqa: E402

# GLM-5.3-Flash schedule (zai-org/GLM-5.3-Flash config.json): every fourth
# layer is MLA with the sparse indexer, the rest are KDA; the first three
# layers use a dense MLP and the rest use MoE.
NUM_LAYERS = 45
FIRST_MLA_LAYER = 3
FIRST_MOE_LAYER = 3
MHC_STREAMS = 4
MHC_PARAMETERS = {
    "hc_attn_fn",
    "hc_attn_base",
    "hc_attn_scale",
    "hc_ffn_fn",
    "hc_ffn_base",
    "hc_ffn_scale",
}

ATTENTION_ROLE = frozenset(("attention",))
FFN_ROLE = frozenset(("ffn",))
NO_ROLES: frozenset[str] = frozenset()
# Checkpoint paths as stored in zai-org/GLM-5.3-Flash (model.safetensors.index.json).
LAYER = "model.language_model.layers"
WEIGHT_ROLE_CASES = (
    ("embedding", "model.language_model.embed_tokens.weight", ATTENTION_ROLE),
    ("final-norm", "model.language_model.norm.weight", ATTENTION_ROLE),
    ("lm-head", "lm_head.weight", ATTENTION_ROLE),
    ("vision-tower", "model.visual.blocks.0.attn.qkv.weight", NO_ROLES),
    ("kda-projection", f"{LAYER}.0.self_attn.q_proj.weight", ATTENTION_ROLE),
    ("kda-state-parameter", f"{LAYER}.0.self_attn.A_log", ATTENTION_ROLE),
    (
        "mla-projection-scale",
        f"{LAYER}.3.self_attn.q_a_proj.weight_scale_inv",
        ATTENTION_ROLE,
    ),
    ("sparse-indexer", f"{LAYER}.3.self_attn.indexer.wk.weight", ATTENTION_ROLE),
    ("input-layernorm", f"{LAYER}.0.input_layernorm.weight", ATTENTION_ROLE),
    (
        "post-attention-layernorm",
        f"{LAYER}.0.post_attention_layernorm.weight",
        ATTENTION_ROLE,
    ),
    ("mhc-attention-mix", f"{LAYER}.0.hc_attn_fn", ATTENTION_ROLE),
    ("mhc-ffn-mix", f"{LAYER}.3.hc_ffn_scale", ATTENTION_ROLE),
    ("dense-mlp-projection", f"{LAYER}.0.mlp.gate_proj.weight", FFN_ROLE),
    ("moe-router", f"{LAYER}.3.mlp.gate.weight", FFN_ROLE),
    ("moe-router-bias", f"{LAYER}.3.mlp.gate.e_score_correction_bias", FFN_ROLE),
    (
        "routed-expert-scale",
        f"{LAYER}.3.mlp.experts.0.down_proj.weight_scale_inv",
        FFN_ROLE,
    ),
    ("shared-expert", f"{LAYER}.3.mlp.shared_experts.up_proj.weight", FFN_ROLE),
)


class _FakeMissingLayer(nn.Module):
    pass


class _FakeMhcOp(nn.Module):
    pass


class _FakeGlm5NextTextConfig(SimpleNamespace):
    def is_kda_layer(self, layer_idx: int) -> bool:
        return (
            layer_idx < len(self.layer_types)
            and self.layer_types[layer_idx] == "linear_attention"
        )


def _text_config() -> _FakeGlm5NextTextConfig:
    return _FakeGlm5NextTextConfig(
        hc_eps=1e-6,
        hidden_act="silu",
        hidden_size=16,
        index_kpool=4,
        index_topk=None,
        intermediate_size=32,
        is_moe=True,
        kv_lora_rank=8,
        layer_types=[
            "deepseek_sparse_attention"
            if layer_idx % 4 == FIRST_MLA_LAYER
            else "linear_attention"
            for layer_idx in range(NUM_LAYERS)
        ],
        max_position_embeddings=4096,
        mhc=True,
        mhc_num_residual_streams=MHC_STREAMS,
        mhc_post_mult_value=2.0,
        mhc_sinkhorn_iterations=20,
        mhc_tau=1.0,
        mla_nope=True,
        mlp_layer_types=["dense"] * FIRST_MOE_LAYER
        + ["sparse"] * (NUM_LAYERS - FIRST_MOE_LAYER),
        n_routed_experts=8,
        num_attention_heads=2,
        num_hidden_layers=NUM_LAYERS,
        q_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        rms_norm_eps=1e-5,
        swiglu_limit=None,
        v_head_dim=8,
        vocab_size=128,
    )


def _vllm_config(*, language_model_only: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=None,
        kernel_config=SimpleNamespace(enable_jit_warmup=False),
        lora_config=None,
        model_config=SimpleNamespace(
            hf_config=_text_config(),
            multimodal_config=SimpleNamespace(
                language_model_only=language_model_only,
            ),
        ),
        parallel_config=SimpleNamespace(
            enable_eplb=False,
            pipeline_parallel_size=1,
            use_sequence_parallel_moe=False,
        ),
        quant_config=None,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        speculative_config=None,
    )


@pytest.fixture
def construction_env(monkeypatch: pytest.MonkeyPatch):
    # The shared stage double records constructor prefixes; these tests check
    # the constructed module kinds and parameters instead.
    calls: dict[str, list[str]] = {
        "kda": [],
        "mla": [],
        "dense": [],
        "moe": [],
        "norm": [],
    }
    role = ["attention"]

    def bind(stage_type):
        # A bound subclass keeps the double a type, as the native decoder
        # checks ``isinstance(self.mlp, Glm5NextMoE)``.
        def init(self, *args, **kwargs) -> None:
            stage_type.__init__(self, calls, *args, **kwargs)

        return type(stage_type.__name__, (stage_type,), {"__init__": init})

    monkeypatch.setattr(
        adapter.native,
        "Glm5NextLinearAttention",
        bind(_stage_type("kda")),
    )
    monkeypatch.setattr(
        adapter.native,
        "Glm5NextMLAAttention",
        bind(_stage_type("mla")),
    )
    monkeypatch.setattr(adapter.native, "Glm5NextMLP", bind(_stage_type("dense")))
    monkeypatch.setattr(adapter.native, "Glm5NextMoE", bind(_stage_type("moe")))
    monkeypatch.setattr(adapter.native, "RMSNorm", bind(_stage_type("norm")))
    monkeypatch.setattr(adapter.native, "PPMissingLayer", _FakeMissingLayer)
    for mhc_op in ("MHCPreOp", "MHCPostOp", "MHCFusedPostPreOp"):
        monkeypatch.setattr(adapter.native, mhc_op, _FakeMhcOp)
    monkeypatch.setattr(
        adapter,
        "parse_afd_config",
        lambda *_args, **_kwargs: AFDConfig(role=role[0]),
    )
    return SimpleNamespace(role=role)


def _make_layer(construction_env, *, role: str, layer_idx: int):
    construction_env.role[0] = role
    vllm_config = _vllm_config()
    return adapter.AFDGlm5NextDecoderLayer(
        vllm_config,
        vllm_config.model_config.hf_config,
        layer_idx,
        prefix=f"model.layers.{layer_idx}",
    )


@pytest.mark.parametrize(
    ("layer_idx", "attention_kind"),
    [(0, "kda"), (FIRST_MLA_LAYER, "mla")],
    ids=["kda-dense-layer", "mla-moe-layer"],
)
def test_attention_role_constructs_attention_norms_and_mhc_only(
    construction_env,
    layer_idx: int,
    attention_kind: str,
) -> None:
    layer = _make_layer(construction_env, role="attention", layer_idx=layer_idx)

    assert layer.self_attn.kind == attention_kind
    assert layer.input_layernorm.kind == "norm"
    assert layer.post_attention_layernorm.kind == "norm"
    assert isinstance(layer.mlp, adapter.RemoteFFNProxy)
    assert layer.mlp.layer_idx == layer_idx
    assert set(dict(layer.named_parameters())) == {
        "self_attn.weight",
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        *MHC_PARAMETERS,
    }


@pytest.mark.parametrize(
    ("layer_idx", "ffn_kind"),
    [(0, "dense"), (FIRST_MOE_LAYER, "moe")],
    ids=["dense-layer", "moe-layer"],
)
def test_ffn_role_constructs_only_native_ffn_block(
    construction_env,
    layer_idx: int,
    ffn_kind: str,
) -> None:
    layer = _make_layer(construction_env, role="ffn", layer_idx=layer_idx)

    assert layer.mlp.kind == ffn_kind
    assert isinstance(layer.self_attn, _FakeMissingLayer)
    assert isinstance(layer.input_layernorm, _FakeMissingLayer)
    assert isinstance(layer.post_attention_layernorm, _FakeMissingLayer)
    assert set(dict(layer.named_parameters())) == {"mlp.weight"}


def test_ffn_compute_delegates_to_native_ffn_block(construction_env) -> None:
    layer = _make_layer(construction_env, role="ffn", layer_idx=FIRST_MOE_LAYER)
    expected = torch.full((2, 4), 7.0)
    layer.mlp.forward = lambda hidden_states: expected

    output = layer.compute_ffn_output(torch.zeros(2, 4))

    assert output is expected


def test_attention_role_rejects_local_ffn_compute(construction_env) -> None:
    layer = _make_layer(construction_env, role="attention", layer_idx=0)

    with pytest.raises(RuntimeError, match="requires the AFD FFN role"):
        layer.compute_ffn_output(torch.zeros(2, 4))


@pytest.fixture
def model_env(monkeypatch: pytest.MonkeyPatch, construction_env):
    class FakeStage(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(1))

    def fake_make_layers(num_layers, layer_fn, prefix):
        layers = nn.ModuleList(
            layer_fn(f"{prefix}.{layer_idx}") for layer_idx in range(num_layers)
        )
        return 0, num_layers, layers

    monkeypatch.setattr(adapter.native, "VocabParallelEmbedding", FakeStage)
    monkeypatch.setattr(adapter.native, "make_layers", fake_make_layers)
    monkeypatch.setattr(
        adapter.native,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    monkeypatch.setattr(
        adapter.native,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    return construction_env


@pytest.mark.parametrize(
    ("role", "owns_lifecycle_modules"),
    [("attention", True), ("ffn", False)],
)
def test_model_allocates_embedding_and_final_norm_only_on_attention(
    model_env,
    role: str,
    owns_lifecycle_modules: bool,
) -> None:
    model_env.role[0] = role

    model = adapter.AFDGlm5NextModel(vllm_config=_vllm_config())

    assert (
        not isinstance(model.embed_tokens, _FakeMissingLayer)
    ) is owns_lifecycle_modules
    assert (not isinstance(model.norm, _FakeMissingLayer)) is owns_lifecycle_modules
    assert len(model.layers) == NUM_LAYERS


def test_model_reports_config_driven_moe_layer_indices() -> None:
    model = object.__new__(adapter.AFDGlm5NextModel)
    object.__setattr__(model, "config", _text_config())

    assert model.get_experts_layer_indices() == tuple(
        range(FIRST_MOE_LAYER, NUM_LAYERS),
    )


@pytest.mark.parametrize(
    ("is_cuda", "afd_config", "overrides", "message"),
    [
        (False, AFDConfig(), {}, "CUDA only"),
        (
            True,
            AFDConfig(compute_gate_on_attention=True),
            {},
            "compute_gate_on_attention=false",
        ),
        (
            True,
            AFDConfig(),
            {"use_sequence_parallel_moe": True},
            "sequence-parallel MoE",
        ),
        (True, AFDConfig(), {"enable_eplb": True}, "EPLB"),
        (
            True,
            AFDConfig(),
            {"pipeline_parallel_size": 2},
            "pipeline parallelism",
        ),
    ],
)
def test_unsupported_parallel_modes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    afd_config: AFDConfig,
    overrides: dict[str, int | bool],
    message: str,
) -> None:
    vllm_config = _vllm_config()
    for name, value in overrides.items():
        setattr(vllm_config.parallel_config, name, value)
    monkeypatch.setattr(
        adapter,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: is_cuda),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._validate_supported_config(vllm_config, afd_config)


@pytest.mark.parametrize(
    ("config_name", "message"),
    [
        ("speculative_config", "speculative decoding"),
        ("lora_config", "LoRA"),
    ],
)
def test_unsupported_model_features_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    message: str,
) -> None:
    vllm_config = _vllm_config()
    setattr(vllm_config, config_name, SimpleNamespace())
    monkeypatch.setattr(
        adapter,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )

    with pytest.raises(RuntimeError, match=message):
        adapter._validate_supported_config(vllm_config, AFDConfig())


def test_multimodal_execution_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter,
        "current_platform",
        SimpleNamespace(is_cuda=lambda: True),
    )

    with pytest.raises(RuntimeError, match="--language-model-only"):
        adapter._validate_supported_config(
            _vllm_config(language_model_only=False),
            AFDConfig(),
        )


@pytest.mark.parametrize(
    ("checkpoint_name", "roles"),
    [case[1:] for case in WEIGHT_ROLE_CASES],
    ids=[case[0] for case in WEIGHT_ROLE_CASES],
)
def test_weight_role_policy(
    checkpoint_name: str,
    roles: frozenset[str],
) -> None:
    assert adapter._checkpoint_weight_roles(checkpoint_name) == roles


@pytest.mark.parametrize(
    ("role", "expected_names"),
    [
        (
            "attention",
            [
                "model.language_model.embed_tokens.weight",
                f"{LAYER}.3.self_attn.kv_b_proj.weight",
                f"{LAYER}.3.hc_attn_base",
                "lm_head.weight",
            ],
        ),
        (
            "ffn",
            [
                f"{LAYER}.3.mlp.gate.weight",
                f"{LAYER}.3.mlp.experts.7.gate_proj.weight",
            ],
        ),
    ],
)
def test_load_weights_filters_once_and_delegates_to_native_loader(
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    expected_names: list[str],
) -> None:
    names = [
        "model.language_model.embed_tokens.weight",
        "model.visual.patch_embed.proj.weight",
        f"{LAYER}.3.self_attn.kv_b_proj.weight",
        f"{LAYER}.3.hc_attn_base",
        f"{LAYER}.3.mlp.gate.weight",
        f"{LAYER}.3.mlp.experts.7.gate_proj.weight",
        "lm_head.weight",
    ]
    weights = _OneShotWeights(names)
    seen: list[tuple[str, Tensor]] = []
    native_result = {"native.loaded_params"}

    def fake_native_loader(self, filtered_weights):
        assert iter(filtered_weights) is filtered_weights
        seen.extend(filtered_weights)
        return native_result

    monkeypatch.setattr(
        adapter.native.Glm4vForConditionalGeneration,
        "load_weights",
        fake_native_loader,
    )
    model = object.__new__(adapter.AFDGlm5NextForConditionalGeneration)
    object.__setattr__(model, "afd_role", role)

    result = model.load_weights(weights)

    assert result is native_result
    assert [name for name, _tensor in seen] == expected_names
    assert [tensor for _name, tensor in seen] == [
        weights.items[names.index(name)][1] for name in expected_names
    ]
    assert weights.iterations == 1
