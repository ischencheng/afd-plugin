# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from torch import Tensor

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from afd_plugin.model_executor.models import glm5_next as adapter  # noqa: E402
from tests.unit.fakes.weights import OneShotWeights as _OneShotWeights  # noqa: E402

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


def test_native_text_loader_and_mapping_remain_native_owned() -> None:
    assert "forward" not in adapter.AFDGlm5NextModel.__dict__
    assert "load_weights" not in adapter.AFDGlm5NextModel.__dict__
    assert "load_weights" not in adapter.AFDGlm5NextForCausalLM.__dict__
    assert (
        adapter.AFDGlm5NextForConditionalGeneration.hf_to_vllm_mapper
        is adapter.native.Glm5NextForConditionalGeneration.hf_to_vllm_mapper
    )
