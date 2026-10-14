# MRV2 Async CAM FFN FULL graph

AFD #449 adds one complete routed-expert transaction to an NPU graph:
dispatch receive (DR), existing layered W4A8 FFN, and combine send (CS).
Attention remains MRV2 eager. This path does not complete the wider MRV2
migration in #451.

## FULL selection and startup

The initial target is the pinned vLLM 0.30 / vLLM-Ascend stack on A3/910C,
with homogeneous DeepSeek-V4 W4A8 layers, dynamicQuant=1, static expert
placement, F TP=1, and Attention-side gating. Both roles select
`VLLM_USE_V2_MODEL_RUNNER=1`. Attention keeps `--enforce-eager`.
Native DBO and its `async_moe_ubatching` alias must be disabled on every rank.

On F, set `AFD_ASYNC_CAM_LAYERED_GMM=1`, omit `--enforce-eager`, and pass
`--compilation-config '{"cudagraph_mode":"FULL"}'`. Piecewise and decode-only
modes are rejected. `BATCH_SIZE_FACTOR` determines the existing native receive
capacity; it must cover the dispatched TP-group token envelope. Per-source
indices must fit the compact protocol's uint16 bound.

The runner prepares its executor and advertises a static plan through the
existing rendezvous Store before HCCL initialization. Every F must request the
same mode. FULL is strict by default. Setting connector extra configuration
`ffn_graph_allow_eager_fallback: true` on every F permits a common eager choice
when graph executor/signature support is unavailable. Invalid transport
configuration still fails. Logs record requested/effective mode and the reason.
There is no fallback once capture or execution has started.

FULL startup consists of:

1. Validate matching transport and executor layouts across ranks.
2. Run an ordinary paired eager transaction using the original DS/DR/CS/CR
   operators. Every A rank participates concurrently in this first MC2 call.
3. Capture one DR → layered FFN → CS graph on each F.
4. Replay with the warmup input and require exact agreement with that eager
   result. Every A waits for the F service loops to announce readiness.

There is no separate initialization operator. Startup uses known transactions
and Store phase barriers; business requests have no added cross-DP barrier.
Fresh-process validation covers this warmup and capture sequence on the pinned
backend, including two independent Attention sources.

## Capture boundary and ownership

The Async FFN runner owns one graph, private allocator pool and execution stream. Each
business batch queues as many replays as there are loaded FFN layers, followed
by one slot D2H copy and a completion fence. This bounded submission count is
not a request boundary: every DR independently selects ready work. Replays on
the same stream reuse stable buffers in order; dynamic layers and expert counts
stay on device. Startup self-checks still submit one replay at a time.
The service thread waits once per batch with device synchronization. A partially
filled batch may wait for more work after A has received the last valid CS;
independent STOP publication lets all queued replays drain during shutdown.
The existing worker control thread checks for drain independently; ordinary
transactions do not query the Store or acquire the stop-publication lock.

| Resource | Owner |
| --- | --- |
| DR activations, scales, group counts and original return metadata | Runner's retained connector payload/context |
| Device layer slot, captured FFN intermediates and output | Runner |
| CS output placeholder | Retained transfer state |
| STOP/error control, valid-layer table, pinned STOP source, copy stream/event | Connector |
| Weights and encoded scale/compensation lists | Runner's model and executor |
| Captured native workspace | Graph's private allocator pool |
| HCCL process group, communication arguments and windows | Connector |

The MRV2 F runner composes model loading, the existing eager executor and graph
execution. The connector owns CAM metadata and transport. It uses no new
generic connector capability or work-item framework. Main's layered arithmetic,
ordinary eager decoding and eager synchronization policy are preserved.

There is no graph cache, online weight replacement or partial-rank restart.
Changing weights, layer layout, capacity or topology requires fresh workers.

## Native interface and normal stopping

The four existing torch/ACLNN communication interfaces remain available. One
new graph DR entry point, `afd_async_dispatch_recv_controlled`, accepts:

- `stop[2]`, initially zero: host-written STOP and device-written first error;
- `valid_layers[N]`, the immutable int64 table matching the executor's weights.

It returns the original four DR tensors plus a true one-element int64
`layer_slot` output. Nonnegative values select a layer. Negative values mean
STOP (-1), invalid layer (-2), capacity error (-3), or insufficient UB (-4).
AIV0 selects and publishes the decision; all AIVs cross the same barriers.
Terminal decisions clear local group counts and write a negative layer in the
local return metadata. The normal CAM wire header is unchanged.
The first error persists across queued replays and takes precedence over STOP.
Only initialization clears both control words; STOP copies write the first word
alone. Later replays skip receiving, computing and sending after an error.

Layered FFN exits before indexing weights on a negative slot. Original CS
recognizes a negative local layer and skips peer writes. A valid zero-row
transaction keeps a nonnegative layer and still sends normal completion.
There is no controlled CS, sequence counter or device epoch protocol.

Normal stopping applies to this new FULL service:

1. After requests finish, each A verifies its original CR owners consumed all
   pending transfers, fences device work once and publishes its drain proof.
2. Existing F control health checks poll for every A proof independently of the
   service thread's completion wait. They submit a retained pinned-host STOP
   copy on a separate stream. The stop RPC takes over polling during shutdown.
3. Blocking DR exits. Every remaining queued replay sees STOP and skips FFN/CS.
   F fences the entire batch and STOP copy before its final acknowledgement.
4. A retains transport through those acknowledgements. Each A publishes its
   final Store access; the Store owner waits for all A publications.
5. The F stop RPC joins the service thread, resets the graph, then releases
   transport and model resources. STOP publication and terminal acknowledgement
   share a local lock; no control poll accesses Store after STOP publication.

Completion and shutdown calls retain resources on a failed launch, native
error or timeout. They do not replay a partly executed graph or silently run
eager instead. Forced process termination does not count as normal stopping.
Existing eager shutdown behavior is outside this change.

## Existing computation semantics

FULL uses the existing layered executor without adding activation clipping.
`swiglu_limit` remains metadata; layered W13 preserves main's existing behavior.
Compare FULL against corresponding layered eager with identical inputs and
weights. Ordinary non-layered eager may apply the limit differently; this
change does not repair that pre-existing difference or upstream quantization
and checkpoint-loader limitations.

## Validation

Use the pinned vLLM/vLLM-Ascend stack and matching native package. The real-model
fixture targets DeepSeek-V4 Flash per-channel W4A8 on A3/910C with 4A/4F,
Attention TP4, FFN TP1, receive capacity 128 and DBO disabled. Synthetic per-group
weights do not qualify loading a real per-group checkpoint. Broader models and
topologies require their own validation.

- Host contracts: `tests/unit/connectors/test_async_cam_startup.py`,
  `tests/unit/v1/worker/test_async_ffn_graph.py`, and
  `tests/unit/v1/worker/test_async_ffn_runtime.py`.
- Native transport/zero-work/STOP/invalid-layer checks:
  `tests/npu/async_cam_control.py`. Start fresh processes with two devices,
  or three for `--zero-rank`.
- Synthetic layered eager/FULL comparison:
  `tests/npu/async_cam_runtime.py`. Run eager with `--save-outputs PREFIX`,
  then FULL with `--reference-outputs PREFIX`, keeping all other arguments
  identical. `--boundary` covers chunking and poisoned unused buffer capacity;
  `--attn-ranks 2` covers changing DP sources. Synthetic per-group weights do
  not qualify loading a real per-group checkpoint.
- Real MRV2 generation: the bounded 4A/4F fixture below. Its eager observations
  add synchronization for functional evidence; it is not a performance test.

```bash
python -m tests.e2e.models.deepseek_v4_flash.ffn_graph \
  --mode eager --model /path/to/checkpoint --afd-host "$AFD_HOST" \
  --devices 0,1,2,3,4,5,6,7 --output-dir /path/to/eager-evidence
python -m tests.e2e.models.deepseek_v4_flash.ffn_graph \
  --mode full --model /path/to/checkpoint --afd-host "$AFD_HOST" \
  --devices 0,1,2,3,4,5,6,7 --output-dir /path/to/full-evidence \
  --reference-dir /path/to/eager-evidence
```

The fixture saves inputs, generated tokens and per-worker execution evidence.
FULL must instantiate the intended runner with all 43 FFN layers, capture once,
replay during requests, and match the saved eager tokens. Batch-end slot samples
do not establish per-layer execution coverage. After requests complete, both roles receive
termination signals before either is awaited. FULL acceptance requires normal
service exits and all workers releasing resources without forced cleanup.
Eager reference cleanup is recorded separately and is not a new graceful-stop
requirement. Failed request/execution runs cannot supply a passing reference.
