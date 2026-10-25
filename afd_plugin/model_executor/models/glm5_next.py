# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""GLM-5.3-Flash AFD wrapper for the NVIDIA ``glm5next`` implementation.

The split is placed at each decoder layer's ``mlp`` call. Attention keeps the
KDA or MLA attention, both norms, and the mHC residual-stream state, and runs
the native decoder forward with a parameter-free ``RemoteFFNProxy`` as
``mlp``. Only the normalized two-dimensional FFN input crosses to FFN, which
runs the complete native dense MLP or ``Glm5NextMoE`` (router, routed and
shared experts) and returns the FFN output.
"""

from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.models.glm5next.nvidia import model as native
from vllm.platforms import current_platform

from afd_plugin.config import AFDConfig, parse_afd_config
from afd_plugin.model_executor.models.deepseek_v2 import RemoteFFNProxy

_ATTENTION_ROLE = frozenset(("attention",))
_FFN_ROLE = frozenset(("ffn",))
_NO_ROLES: frozenset[str] = frozenset()


def _is_moe_layer(config: native.Glm5NextConfig, layer_idx: int) -> bool:
    """Return whether the native GLM-5.3-Flash schedule selects MoE."""
    mlp_layer_types = config.mlp_layer_types
    mlp_type = (
        mlp_layer_types[layer_idx]
        if layer_idx < len(mlp_layer_types)
        else (mlp_layer_types[-1] if mlp_layer_types else "sparse")
    )
    return (
        config.is_moe and config.n_routed_experts is not None and (mlp_type == "sparse")
    )


def _weight_layer_path(name: str) -> tuple[int, str] | None:
    """Return ``(layer index, stage)`` for a native decoder weight path."""
    parts = name.split(".")
    for marker_idx, part in enumerate(parts[:-2]):
        if part != "layers":
            continue
        try:
            layer_idx = int(parts[marker_idx + 1])
        except ValueError:
            continue
        return layer_idx, parts[marker_idx + 2]
    return None


def _checkpoint_weight_roles(name: str) -> frozenset[str]:
    """Classify one native GLM-5.3-Flash checkpoint path by AFD owner."""
    if "visual" in name.split("."):
        return _NO_ROLES
    layer_path = _weight_layer_path(name)
    if layer_path is None:
        # Embeddings, final norm, and lm_head are required only by Attention.
        return _ATTENTION_ROLE
    _layer_idx, stage = layer_path
    if stage == "mlp":
        return _FFN_ROLE
    # Attention, norms, and the per-layer mHC parameters.
    return _ATTENTION_ROLE


def _iter_role_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    role: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Consume a checkpoint iterator once and retain only this role's paths."""
    for name, loaded_weight in weights:
        if role in _checkpoint_weight_roles(name):
            yield name, loaded_weight


def _validate_supported_config(
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
) -> None:
    """Reject GLM-5.3-Flash modes not covered by the initial CUDA contract."""
    parallel_config = vllm_config.parallel_config
    if not current_platform.is_cuda():
        raise RuntimeError("AFD GLM-5.3-Flash currently supports CUDA only")
    if afd_config.compute_gate_on_attention:
        raise RuntimeError(
            "AFD GLM-5.3-Flash requires compute_gate_on_attention=false",
        )
    if parallel_config.use_sequence_parallel_moe:
        raise RuntimeError(
            "AFD GLM-5.3-Flash does not support sequence-parallel MoE",
        )
    if parallel_config.enable_eplb:
        raise RuntimeError("AFD GLM-5.3-Flash does not support EPLB")
    if parallel_config.pipeline_parallel_size != 1:
        raise RuntimeError(
            "AFD GLM-5.3-Flash does not support pipeline parallelism",
        )
    if vllm_config.speculative_config is not None:
        raise RuntimeError(
            "AFD GLM-5.3-Flash does not support speculative decoding",
        )
    if vllm_config.lora_config is not None:
        raise RuntimeError("AFD GLM-5.3-Flash does not support LoRA")
    if vllm_config.model_config.multimodal_config.language_model_only is not True:
        raise RuntimeError(
            "AFD GLM-5.3-Flash currently supports text-only execution only; "
            "pass --language-model-only",
        )


class AFDGlm5NextDecoderLayer(native.Glm5NextDecoderLayer):
    """GLM-5.3-Flash decoder layer with separable Attention and FFN execution."""

    # Patch reason: native Glm5NextDecoderLayer constructs Attention, the dense
    # MLP or MoE, both norms, and the mHC parameters on every rank.
    # Patch functionality: construct only the modules owned by the active AFD
    # role. Attention keeps Attention, norms, and mHC state with a
    # parameter-free RemoteFFNProxy as ``mlp``, so the native forward is reused
    # unchanged; FFN keeps the native dense MLP or Glm5NextMoE.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.30.0, vllm/models/glm5next/nvidia/model.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: native.Glm5NextConfig,
        layer_idx: int,
        prefix: str = "",
        topk_indices_buffer: torch.Tensor | None = None,
        is_mtp_layer: bool = False,
        **kwargs,
    ) -> None:
        # ### PATCH START: initialize the role-aware layer without native allocation.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        # ### PATCH END: initialize the role-aware layer without native allocation.

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.is_moe = config.is_moe
        self.num_hidden_layers = config.num_hidden_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.n_routed_experts
        self.is_mtp_layer = is_mtp_layer
        self.mhc = config.mhc
        is_kda_layer = not is_mtp_layer and config.is_kda_layer(layer_idx)
        self.layer_kind = "kda" if is_kda_layer else "mla"
        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        # ### PATCH START: allocate only the active role's execution stage.
        if self.afd_role == "attention":
            if is_kda_layer:
                self.self_attn = native.Glm5NextLinearAttention(
                    config=config,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.self_attn",
                )
            else:
                # MLA layers require the latent head dims, which are guaranteed
                # set on MLA configs; narrow away the `int | None`.
                assert config.v_head_dim is not None
                assert config.kv_lora_rank is not None
                self.self_attn = native.Glm5NextMLAAttention(
                    vllm_config=vllm_config,
                    config=config,
                    hidden_size=self.hidden_size,
                    num_heads=config.num_attention_heads,
                    qk_nope_head_dim=config.qk_nope_head_dim,
                    qk_rope_head_dim=config.qk_rope_head_dim,
                    v_head_dim=config.v_head_dim,
                    q_lora_rank=config.q_lora_rank,
                    kv_lora_rank=config.kv_lora_rank,
                    max_position_embeddings=config.max_position_embeddings,
                    cache_config=cache_config,
                    quant_config=None,  # MLA projections are BF16 in checkpoint
                    prefix=f"{prefix}.self_attn",
                    topk_indices_buffer=topk_indices_buffer,
                    skip_rope=config.mla_nope,
                )
            self.mlp = RemoteFFNProxy(layer_idx=layer_idx)
            self.input_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = native.RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
        else:
            self.self_attn = native.PPMissingLayer()
            if _is_moe_layer(config, layer_idx):
                self.mlp = native.Glm5NextMoE(
                    config=config,
                    parallel_config=parallel_config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                )
            else:
                self.mlp = native.Glm5NextMLP(
                    hidden_size=self.hidden_size,
                    intermediate_size=config.intermediate_size,
                    hidden_act=config.hidden_act,
                    quant_config=quant_config,
                    prefix=f"{prefix}.mlp",
                    swiglu_limit=config.swiglu_limit,
                )
            self.input_layernorm = native.PPMissingLayer()
            self.post_attention_layernorm = native.PPMissingLayer()
        # ### PATCH END: allocate only the active role's execution stage.

        # Cached for the hot forward path (isinstance per layer per step).
        self._mlp_is_moe = isinstance(self.mlp, native.Glm5NextMoE)
        # In SP, the attention output projection leaves a partial sum; the
        # decoder-layer reduce_scatter after attention completes it (DSv4 pattern).
        # MTP layers use the non-mHC path which has no sp_reduce_scatter, so
        # their o_proj must still reduce normally.
        if self.is_sequence_parallel and not is_mtp_layer:
            self.self_attn.o_proj.reduce_results = False

        # ### PATCH START: mHC state is constructed only on Attention.
        if self.afd_role != "attention":
            return
        # ### PATCH END: mHC state is constructed only on Attention.
        if self.mhc and not is_mtp_layer:
            # mhc config
            self.mhc_num_residual_streams = config.mhc_num_residual_streams
            self.mhc_tau = config.mhc_tau
            self.hc_eps = config.hc_eps
            self.mhc_sinkhorn_iterations = config.mhc_sinkhorn_iterations
            self.mhc_post_mult_value = config.mhc_post_mult_value

            n = config.mhc_num_residual_streams
            d_model = n * self.hidden_size
            mix_hc = (2 + n) * n

            self.n = n

            # attn hc
            self.hc_attn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            # ffn hc
            self.hc_ffn_fn = nn.Parameter(
                torch.empty(mix_hc, d_model, dtype=torch.float32)
            )
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

            self.mhc_pre_op = native.MHCPreOp()
            self.mhc_post_op = native.MHCPostOp()
            self.mhc_fused_post_pre_op = native.MHCFusedPostPreOp()

            if vllm_config.kernel_config.enable_jit_warmup:
                from vllm.model_executor.kernels.mhc.tilelang_kernels import (
                    _HC_PRENORM_GEMM_TILELANG_KERNEL,
                    _MHC_FUSED_TILELANG_KERNEL,
                    _MHC_POST_TILELANG_KERNEL,
                    _MHC_PRE_BIG_FUSE_TILELANG_KERNEL,
                )
                from vllm.utils.deep_gemm import is_deep_gemm_supported

                include_pre_gemm_splits = is_deep_gemm_supported()
                _MHC_PRE_BIG_FUSE_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                    use_norm_weight=True,
                    include_pre_gemm_splits=include_pre_gemm_splits,
                    include_broadcast_splits=False,
                    rms_eps=self.rms_norm_eps,
                    hc_pre_eps=self.hc_eps,
                    hc_sinkhorn_eps=self.hc_eps,
                    hc_post_mult_value=self.mhc_post_mult_value,
                    sinkhorn_repeat=self.mhc_sinkhorn_iterations,
                    norm_eps=(
                        self.input_layernorm.variance_epsilon,
                        self.post_attention_layernorm.variance_epsilon,
                    ),
                )
                if not include_pre_gemm_splits:
                    _HC_PRENORM_GEMM_TILELANG_KERNEL.register_warmup(
                        vllm_config,
                        hidden_size=self.hidden_size,
                        hc_mult=self.n,
                        n_out=self.n * (2 + self.n),
                    )
                _MHC_POST_TILELANG_KERNEL.register_warmup(
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                )
                _MHC_FUSED_TILELANG_KERNEL.register_warmup(
                    vllm_config,
                    hidden_size=self.hidden_size,
                    hc_mult=self.n,
                )

    def compute_ffn_output(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Execute this layer's complete native dense MLP or MoE on FFN."""
        if self.afd_role != "ffn":
            raise RuntimeError(
                "GLM-5.3-Flash FFN compute requires the AFD FFN role",
            )
        return self.mlp(hidden_states)


class AFDGlm5NextModel(native.Glm5NextModel):
    """Native GLM-5.3-Flash model lifecycle with role-aware decoder layers."""

    # Patch reason: native Glm5NextModel builds native decoder layers through a
    # local factory and allocates the sparse-index buffer, embedding, and final
    # norm on every rank.
    # Patch functionality: build role-aware decoder layers and allocate the
    # sparse-index buffer, embedding, and final norm only on Attention. The
    # native forward and load_weights are inherited unchanged.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.30.0, vllm/models/glm5next/nvidia/model.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: initialize the role-aware model without native allocation.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        # ### PATCH END: initialize the role-aware model without native allocation.

        config = vllm_config.model_config.hf_config
        self.config = config

        self.vocab_size = config.vocab_size
        self.device = native.current_platform.device_type

        self.is_v32 = config.index_topk is not None
        # ### PATCH START: allocate the sparse-index buffer only on Attention.
        if self.is_v32 and self.afd_role == "attention":
            topk_tokens = config.index_topk
            assert topk_tokens is not None
            # Reserve room for the incomplete pool tail.
            kpool = config.index_kpool
            assert kpool is not None
            buffer_width = topk_tokens + (kpool - 1 if kpool > 1 else 0)
            # Sparse MLA tiles top-k in 128 columns; padded slots remain masked.
            sparse_topk_block_n = 128
            buffer_width = (
                (buffer_width + sparse_topk_block_n - 1) // sparse_topk_block_n
            ) * sparse_topk_block_n
            topk_indices_buffer = torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                buffer_width,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            # Full-MLA config or FFN role: no topk buffer.
            topk_indices_buffer = None
        # ### PATCH END: allocate the sparse-index buffer only on Attention.

        # ### PATCH START: construct the embedding only on Attention.
        if self.afd_role == "attention" and native.get_pp_group().is_first_rank:
            self.embed_tokens = native.VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = native.PPMissingLayer()
        # ### PATCH END: construct the embedding only on Attention.

        # ### PATCH START: build role-aware decoder layers.
        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            return AFDGlm5NextDecoderLayer(
                vllm_config=vllm_config,
                config=config,
                layer_idx=layer_idx,
                prefix=prefix,
                topk_indices_buffer=topk_indices_buffer,
            )

        # ### PATCH END: build role-aware decoder layers.

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers",
        )
        # The active slice is fixed after construction; cache it so forward
        # doesn't rebuild the slice (a fresh list) every step.
        self._active_layers = self.layers[self.start_layer : self.end_layer]

        # ### PATCH START: construct the final norm only on Attention.
        if self.afd_role == "attention" and native.get_pp_group().is_last_rank:
            self.norm = native.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()
        # ### PATCH END: construct the final norm only on Attention.

        self.is_sequence_parallel = (
            vllm_config.parallel_config.use_sequence_parallel_moe
        )

        world_size = native.get_tensor_model_parallel_world_size()
        assert config.num_attention_heads % world_size == 0, (
            "num_attention_heads must be divisible by world_size"
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.layers[layer_idx].compute_ffn_output(hidden_states)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(
            layer_idx
            for layer_idx in range(self.config.num_hidden_layers)
            if _is_moe_layer(self.config, layer_idx)
        )


class AFDGlm5NextForCausalLM(native.Glm5NextForCausalLM):
    """GLM-5.3-Flash text model shell owning the role-aware native model."""

    # Patch reason: native Glm5NextForCausalLM hard-codes Glm5NextModel and
    # constructs the LM head on every rank.
    # Patch functionality: construct AFDGlm5NextModel and the LM head only on
    # Attention while preserving the inherited forward, logits, state, and
    # loader methods.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.30.0, vllm/models/glm5next/nvidia/model.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: avoid constructing the full native model first.
        nn.Module.__init__(self)
        self.afd_role = parse_afd_config(vllm_config, validate=False).role
        # ### PATCH END: avoid constructing the full native model first.
        self.model_config = vllm_config.model_config
        self.vllm_config = vllm_config
        self.config = self.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.quant_config = quant_config
        # ### PATCH START: inject the role-aware model.
        self.model = AFDGlm5NextModel(
            vllm_config=vllm_config, prefix=native.maybe_prefix(prefix, "model")
        )
        # ### PATCH END: inject the role-aware model.
        # ### PATCH START: construct the LM head only on Attention.
        if self.afd_role == "attention" and native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
        else:
            self.lm_head = native.PPMissingLayer()
        # ### PATCH END: construct the LM head only on Attention.
        self.logits_processor = native.LogitsProcessor(
            self.config.vocab_size, scale=self.config.logit_scale
        )

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()


class AFDGlm5NextForConditionalGeneration(native.Glm5NextForConditionalGeneration):
    """GLM-5.3-Flash checkpoint wrapper for text-only AFD execution."""

    # Patch reason: the native multimodal shell resolves its language model
    # through the registry, which returns the native causal LM.
    # Patch functionality: validate the supported AFD configuration and
    # construct AFDGlm5NextForCausalLM through the same initialize_model path.
    # Tower staging, forward, and multimodal interfaces stay native.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.30.0, vllm/models/glm5next/nvidia/model.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        # ### PATCH START: enforce AFD's supported GLM-5.3-Flash configuration.
        afd_config = parse_afd_config(vllm_config, validate=False)
        _validate_supported_config(vllm_config, afd_config)
        # ### PATCH END: enforce AFD's supported GLM-5.3-Flash configuration.
        super(native.Glm4vForConditionalGeneration, self).__init__()
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )
        # ### PATCH START: retain the AFD role for filtered loading.
        self.afd_role = afd_config.role
        # ### PATCH END: retain the AFD role for filtered loading.

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = native.Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                # Read eps from the VISION sub-config, not the top-level
                # `config.rms_norm_eps`: Glm5NextConfig.__getattribute__ mirrors
                # the latter onto text_config (1e-5), silently ignoring the
                # vision tower's own (1e-6) rms_norm_eps.
                norm_eps=config.vision_config.rms_norm_eps,
                # Vision tower ships BF16 weights in this fp8 checkpoint (no
                # weight_scale_inv for visual.*), so it must NOT inherit the
                # global fp8 quant_config -- doing so incorrectly quantizes
                # the tower
                # and yields NaN image features. Mirrors the MLA/KDA proj
                # pattern (quant_config=None for BF16 submodules).
                quant_config=None,
                prefix=native.maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            # ### PATCH START: construct the role-aware causal LM.
            self.language_model = initialize_model(
                vllm_config=vllm_config.with_hf_config(
                    config.text_config,
                    architectures=["Glm5NextForCausalLM"],
                ),
                prefix=native.maybe_prefix(prefix, "language_model"),
                model_class=AFDGlm5NextForCausalLM,
            )
            # ### PATCH END: construct the role-aware causal LM.

        self.set_moe_parameters()

        # Glm5NextForCausalLM does not implement make_empty_intermediate_tensors,
        # so pipeline parallelism is gated off (consistent with the text-only
        # model) and we intentionally do not alias it here.

    def compute_ffn_output(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        return self.language_model.compute_ffn_output(hidden_states, layer_idx)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.language_model.get_experts_layer_indices()

    # Patch reason: native loading allocates every checkpoint path locally.
    # Patch functionality: filter checkpoint paths to the owning AFD role once,
    # then delegate to the inherited native loader unchanged.
    # Signature: matches upstream; no added parameters.
    # Upstream: vLLM v0.30.0, vllm/model_executor/models/glm4_1v.py
    # Commit: ced6857afa0ea7b2e3f0846a62e1394e90f15607
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # ### PATCH START: load each checkpoint path only on its owning role.
        role_weights = _iter_role_weights(weights, role=self.afd_role)
        # ### PATCH END: load each checkpoint path only on its owning role.
        return super().load_weights(role_weights)


__all__ = ["AFDGlm5NextForConditionalGeneration"]
