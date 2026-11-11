# GMS v0 read-only shadow attach on vLLM 0.31 (stage 1 of the Foundry 0.31 port)

Branch `rvinodkumar/gms-v0-vllm031`, based on 9629d4fbf1 (`rvinodkumar/gms-v0-glm-recovery`). Not pushed.
Date: 2026-11-10. Pod `rp-v031-a`, node `<gpu-node>`, 4x GB200,
image `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0` (vLLM 0.31.0, torch 2.13.0+cu130, flashinfer 0.7.0.post1,
transformers 5.14.1, ai-dynamo 1.6.0; its bundled gpu_memory_service equals base d66fc3f42f).
Raw results (JSON outputs, logs, nvidia-smi snapshots): a local results directory.

## Result

| | status |
|---|---|
| GMS writer (TP4+EP, V2 runner, 0.31 GLM profile) | works; output identical to no-GMS 0.31 |
| Read-only shadow, real KV, attached while writer serves | works; greedy tokens identical to writer and to no-GMS on all 6 prompts |
| Read-only shadow, scratch KV → park → writer killed → wake → serve (the replenishment flow) | works; identical to writer and no-GMS; wake 4.3 s |
| Generic (non-profile) meta reader on 0.31 | both reported `.item()` sites fixed, then fails on a third (`torch.allclose` in NVFP4 MoE post-load); not supported, use the profile |
| Classic (V1) runner on 0.31 | garbage for every prefill that is CUDA-graph captured (≤ max capture size); root cause found (below); not needed |

### Greedy outputs (temperature 0, logprobs=5, one prompt at a time from an empty prefix cache)

Prompts: 4 short (5–14 tokens; 32–64 output tokens) + needle-in-haystack at 7,565 and 29,604 tokens (both answered
correctly). Reference = no-GMS 0.31 engine (`e0b-nogms`).

| engine | identical tokens (6 prompts) | short-prompt max \|Δlogprob\| | long-prompt max \|Δlogprob\| |
|---|---|---|---|
| writer `e1-writer` | 6/6 | 0.000000 | 0.19 / 0.42 |
| shadow, real KV `e2-reader` | 6/6 | 0.000000 | 0.013 / 0.097 |
| shadow, scratch KV, after wake `e3-shadow` | 6/6 (and 6/6 on a second gen) | 0.000000 | 0.008 / 0.38 |
| noise floor: writer vs itself (2nd gen, same engine) | 6/6 | 0.000000 | 0.49 / 0.29 |

Short prompts are bit-identical. Long prefills (7.5K / 29.6K tokens) are not bit-reproducible even within one engine
(same tokens, logprobs move by up to 0.49 nats on some position), so the long-prompt deltas are at the engine's own
noise floor.

Protocol note: the first attempt generated all 6 prompts in one batch. That is not reproducible on 0.31 (the same
writer gave different short-prompt continuations on a second batched call: batch composition / chunked-prefill
co-scheduling changes kernels). All comparisons above use the sequential protocol (`gen` in `engine.py`).

### Shadow attach time and memory (per GPU, all 4 ranks alike)

| | value |
|---|---|
| Weights published by the writer | 106.84 GiB (writer private offset 11.50 GiB) |
| Shadow weight attach (`Model loading took … seconds`, GMS RO import + 0.31 reconstruction) | 5.3–8.0 s, 0.86 GiB private; vs 171 s cold / 67 s warm page cache from the PVC |
| Shadow engine construct total (`LLM()`; includes profile run, two graph captures 34 s + 59 s, KV init) | 155 s (scratch KV) / 161 s (real KV) |
| Shadow private memory, scratch KV, attached | 10.18 GiB (scratch KV: 16.04 GiB VA reserved, 0.50 GiB physical) |
| Shadow private memory, parked (after `sleep(1)`) | 8.94 GiB |
| Shadow private memory, real KV | 26.09 GiB |
| Device total: writer alone / writer + real-KV shadow / writer + scratch shadow / + parked | 134.99 / 164.84 / 148.93 / 147.69 GiB |
| `sleep(1)` / `wake_up()` | 0.22 s / 4.31 s (remap 710 weight allocations, 106.33 GiB; reallocate 16.04 GiB KV) |
| Device total after the writer died and the shadow woke | 134.26 GiB |

The GMS server keeps the committed weights after the writer dies, so a shadow can also attach with no writer alive
(`e2`, `e5`–`e12` did).

## Code changes (commit e6f725da95 + the NOTES/scripts commit)

`lib/gpu_memory_service/integrations/vllm/glm_reconstruction.py`
- Profiles: `glm52-nvfp4-vllm0271-gb200-tp4-ep4` (unchanged behaviour) and new `glm52-nvfp4-vllm031-gb200-tp4-ep4`
  (vLLM 0.31.0, transformers 5.14.1; CUDA graphs allowed, `enforce_eager`/`cudagraph_mode` part of the identity;
  max_model_len ≤ 40960, qualified here with 29.6K-token prompts). `selected_profile` returns the profile name.
- 0.31 restore: construct on meta (expert maps on the real device), attach the published tensors as before, then run
  each quantization method's `process_weights_after_loading` under vLLM 0.31's `weights_already_processed()` — vLLM's
  own contract for loading post-processed weights (used by its weight-cache IPC loader). That rebuilds the NVFP4 MoE
  kernel (`_restore_padded_moe_dims` + `_build_moe_kernel`, FLASHINFER_TRTLLM) and the KV-scale floats/CPU copies from
  the attached `_k_scale/_v_scale/_q_scale`, with no repacking and no writes into the read-only weight mapping.
  Attention post-load (W_UK_T/W_UV derivation, which would `copy_` into the RO mapping via `replace_parameter(...,
  prefer_copy=True)`) is skipped; those tensors are attached.
- Checks added: expert-map digest per MoE layer (writer-published), rebuilt KV floats must equal the writer's, an
  audit of every simple-typed attribute of every module and quant method (published under
  `__gms__.glm_reconstruction.audit`; result on GPU: "all simple attributes match" on all ranks), a meta-tensor scan
  of plain attributes (empty placeholders such as `attn.kv_cache` are moved to the device), and identity-mismatch
  errors that name the differing keys.
- Accepted model revisions: 53e0691e… and the hub snapshot aec724e8… it is a projection of.

`lib/gpu_memory_service/integrations/vllm/model_loader.py` (generic meta reader, no profile)
- `.item()` #1, `ExpertMapManager` EP-map log line during construction: the map is a pure function of
  (ep_size, ep_rank, placement); `_create_meta_model(..., device_index)` builds it on the device (the same
  `construction_context` the profile uses) and GMS materialization then overwrites it with the published buffer.
- `.item()` #2, KV-cache `process_weights_after_loading` (`if layer.k_scale > 0.0 …`): on meta layers it is deferred:
  the checkpoint placeholders are dropped (as the writer's post-load did) and, after `materialize_module_from_gms`,
  the method runs under `weights_already_processed()`, reading `_k_scale/_v_scale/_q_scale` from the GMS tensors.
- Then the generic path fails on `.item()` #3 (`modelopt.py:986`, `torch.allclose(w13_weight_scale_2[:,0], …)` in
  `ModelOptNvFp4FusedMoE.process_weights_after_loading` on meta; log `results/logs/e5-generic.log`). Replaying
  packing transforms on meta is the wrong model for 0.31; the profile path (attach, then pre-processed post-load) is
  the supported 0.31 reader. Making the generic path do the same needs the writer-published schema (the descriptor),
  i.e. it becomes the profile path.

Tests: `tests/test_gms_reconstruction.py` +3 (0.31 profile admission, unconfigured profile, deferred KV scales).
In the 0.31 image: `20 passed` for `test_gms_reconstruction.py test_component_reconstruction.py`.

## Classic (V1) model runner on 0.31: why GLM-5.2 gives garbage

Root cause: for `GlmMoeDsaForCausalLM` vLLM 0.31 auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` (compilation mode NONE).
PIECEWISE graphs are then captured over the whole eager forward, with an eager break only at
`DeepseekV32Attention._sparse_indexer_and_attn`. The MLA KV-cache and indexer K-cache writes happen *before* that
break, inside `fused_norm_rope(..., slot_mapping=mla_slot, mla_kv_cache=...)`, and whether they happen at all is a
Python decision made at capture time (`if forward_context.attn_metadata is None: mla_kv_cache = None; mla_slot = None`,
`models/deepseek_v32/attention.py` ~347). The classic runner captures PIECEWISE graphs without attention metadata
(`_warmup_and_capture`: `force_attention = cudagraph_runtime_mode == CUDAGraphMode.FULL`,
`v1/worker/gpu_model_runner.py:6874`). So every captured PIECEWISE (prefill) graph omits the KV writes: prompts whose
prefill replays a PIECEWISE graph attend over stale KV. The V2 runner captures breakable PIECEWISE graphs with fresh
attention metadata (`create_forward_fn(desc, warmup=False)`, `v1/worker/gpu/cudagraph_utils.py` ~455), so it is correct.
Under torch.compile piecewise (0.27.1) the KV write lived inside the attention splitting op, outside the graphs.

Evidence (all classic-runner shadows attached through GMS, `VLLM_USE_V2_MODEL_RUNNER=0`):
- `e6`/`e8`/`e10`: prompts of 5, 11, 13, 14, 138, 486 tokens → garbage; 1388, 1722, 2091, 7565, 29604 tokens →
  correct (identical tokens to V2). The default max capture size is 1024. Decode (FULL graphs, captured with
  metadata) is fine: long prompts decode correctly.
- Cross-request contamination: the 138-token prompt's garbage contains `nectar-96`, the answer of the previous
  (2091-token) request, i.e. stale KV in reused slots.
- `e7`: `attention_config.sparse_mla_force_mqa=true` (disables dense-MHA short prefill) → same garbage. Not the
  dense-MHA path.
- `e11`: breakable debug address check (`VLLM_LOGGING_LEVEL=DEBUG`) → no input-address change; garbage differs run to
  run (reads stale memory).
- `e12`: `cudagraph_capture_sizes=[1,2,4,8]` → only the 5-token prompt (padded to 8, captured) is garbage; 11–14,
  138, 486-token prompts become coherent.

Is the classic runner needed? No. The GMS writer and the shadow (attach, scratch KV, sleep/wake) all work on the V2
runner, which is 0.31's default (0.31 calls MRV1 deprecated). If it were ever needed: `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
(falls back to FULL_DECODE_ONLY; untested here) or force attention metadata for breakable PIECEWISE capture in
`_warmup_and_capture`. Consequence for Foundry: SAVE/LOAD must target the V2 runner's capture paths (stage 2).

## Stage 2 (report only): what Foundry SAVE/LOAD must change for 0.31

`pip show vllm`: 0.27.1 image `vllm/vllm-openai@sha256:1c8e60a0…` = vllm 0.27.1, flashinfer 0.6.16.post3,
transformers 5.15.0; 0.31 image = vllm 0.31.0, flashinfer 0.7.0.post1, transformers 5.14.1 (torch 2.13.0+cu130 in both).
Sources compared: `…/tmp/v031/src/{v0271,v031}/vllm`.

Bottom line: on 0.31 none of the current capture hooks fire for GLM. The V2 runner captures FULL graphs itself
(`v1/worker/gpu/cudagraph_utils.py` `CudaGraphManager.capture`, `torch.cuda.graph(...)` ~464-488) and PIECEWISE graphs
through `BreakableCUDAGraphWrapper`/`BreakableCUDAGraphCapture` (`compilation/breakable_cudagraph.py`). Compilation mode
is NONE, so `CUDAGraphWrapper.__call__`, `PiecewiseBackend`, `GPUModelRunner._capture_cudagraphs` and the dispatcher
are not on the path. The hooks' `VLLM_USE_V2_MODEL_RUNNER=0` assumption selects the broken classic runner.

Capture/dispatch, 0.27.1 classic → 0.31 V2:
- Entry: `capture_model → _capture_cudagraphs → _warmup_and_capture → _dummy_run → CUDAGraphWrapper` →
  `capture_model(profile_only=False) → ModelCudaGraphManager.capture → CudaGraphManager.capture`.
- Graph ids: `(mode, BatchDescriptor, piecewise index)` → FULL keyed by `BatchExecutionDescriptor` (uniform decode:
  num_reqs, uniform_token_count); PIECEWISE keyed by `BatchDescriptor(num_tokens, has_lora, num_active_loras)` +
  segment index. Runtime: `dispatch()` → `run_fullgraph` (`graph.replay()`) or `run_pw_graph` (breakable replay).
- Graphs per size: 79 PIECEWISE (FX split) + 1 FULL → 1 FULL, and per PIECEWISE size 79 graph segments interleaved
  with 78 Python eager closures (`fn(*weak_args, **weak_kwargs)`, one per layer, reading the forward context at
  replay). Confirmed on GPU: `BreakableCUDAGraphCapture(graphs=79, …)`. Capture sizes 1..1024 by default (83 sizes).
- Warmups: `cudagraph_num_of_warmups` is ignored by V2; one eager warmup per descriptor is hard-coded, and FULL and
  breakable captures rebuild attention metadata (`create_forward_fn(warmup=False)`).
- Memory profiling: `profile_cudagraph_memory` calls the real `capture_model(profile_only=True)` against a throwaway
  pool with minimal KV (2 FULL samples), then tears down; it runs unless `kv_cache_memory_bytes` is set. (GPU log: a
  34 s profiling capture precedes the 59 s final capture.)
- Baked-in buffers: V2 `InputBuffers` (input_ids, positions, is_padding, query_start_loc, seq_lens), `BlockTables`
  (input block tables, slot mappings, pointer tables; created inside `initialize_kv_cache`), persistent builder
  metadata from `prepare_inputs_to_capture`, FULL output `manager.hidden_states` (allocated at first warmup).
- Pool: one global graph pool shared by the FULL manager and the breakable wrapper.

Change list, in order of necessity:
SAVE
1. Hook FULL capture by re-deriving `CudaGraphManager.capture` (replace the `torch.cuda.graph` stanza with
   capture-or-load); key archives by `BatchExecutionDescriptor` (new filename schema).
2. Hook breakable PIECEWISE capture: per segment in `BreakableCUDAGraphCapture._begin_segment/_end_segment`
   (Foundry graph capture + save), and archive each eager closure (module path + bound method, tensor leaves as
   ptr/shape/stride/dtype, non-tensor args). Foundry's SGLang integration (`integration/sglang/bcg_ops.py`) already
   does this for SGLang's breakable graphs; port it with a bound-method resolver.
3. Accept `capture_model(profile_only=...)` (the 0.27.1 wrapper `patched(self)` raises TypeError); skip saving when
   profiling, or set `kv_cache_memory_bytes` so profiling never runs on SAVE and LOAD alike.
4. Record `manager.hidden_states`, `InputBuffers`, `BlockTables` and persistent builder buffers in the manifest.
5. Re-base the `EngineCore._initialize_kv_caches` re-derivation on 0.31 (`resolve_kv_cache_layout`/
   `set_kv_cache_layout` before profiling, `update_kv_cache_capacity`), or stop overriding it.
6. Make `kernel_warmup` (moved; now runs JIT registry, sparse-MLA autotune, DeepGEMM warmup) and V2 `warmup_kernels`
   (real `execute_model` steps, not patched today) symmetric between SAVE and LOAD.
LOAD
1. FULL: still call `create_forward_fn(desc, warmup=False)` (keeps the allocation cursor trajectory), skip warmup and
   forward, install the loaded graph in `self.graphs[desc]`, rebuild `hidden_states` at the archived address.
2. Breakable: rebuild each entry (`entry.capture.segments` = loaded segment replays + synthesized eager closures,
   `entry.output` from the archived pointer).
3. Drop the `prepare_communication_buffer_for_model` patch (removed in 0.31); patch
   `vllm.v1.worker.gpu.model_runner.get_model_loader` instead of `gpu_model_runner.get_model_loader`.
4. Rewrite the pre-KV mirrors (minimal metadata, input-batch preinit, capture-workspace prewarm) against V2 objects
   (`_init_minimal_kv_cache_for_profiling(runner)`, `_teardown_profiling_state`, `InputBuffers`, `BlockTables`).
5. Lean-load specifics: `graph_capture(device, graph_capture_context=)` no longer takes a stream (route the pinned
   stream through a replacement bound into `vllm.v1.worker.gpu.cudagraph_utils`); `_dummy_run` has a new signature
   (`is_profile` keyword; the `_is_profile_call` match never fires); retarget `fi_ar_early` to the
   `_can_use_flashinfer` call site; `flashinfer_mla_sparse._get_workspace_buffer(device, min_bytes)` reallocates when
   too small, so precreate at the max size.
Collectives (largest design change)
- 0.31 defaults `VLLM_ALLREDUCE_USE_FLASHINFER=1` (0 in 0.27.1; verified `envs.py:1915` vs `:1789`), and the DSA decoder
  calls flashinfer's trtllm fused all-reduce+RMSNorm directly (`models/common/ops/fused_allreduce_rms_norm.py`).
  Decode FULL graphs and small prefill segments will carry mnnvl/Lamport kernels (fabric unicast peer buffers,
  multicast pointer, Lamport flags) instead of vLLM custom AR rows; custom AR/symm-mem only above the ~32 MB budget.
- So the custom-AR row rebinding and symm-mem repin chain may drop out, and a new "mnnvl rebind" contract is needed
  (same VAs for peer/multicast mappings at LOAD, consistent Lamport state at first replay). The FI workspace is a lazy
  global created by the first caller; it must be created at the same point on SAVE and LOAD, before LOAD maps the
  arena (the lean-v5 `fi_ar_early` lesson). Alternative: keep 0.27.1-style collectives
  (`VLLM_ALLREDUCE_USE_FLASHINFER=0` covers only the standalone path; the direct fused call needs a code patch).
Risks
- Python eager segments per layer at replay; weak-ref closures need exact SAVE addresses for every activation in the
  pool; V2 UVA pinned-host buffers live outside the arena; every sha pin of `gpu_model_runner.py` breaks.
- Smaller first port: `VLLM_USE_BREAKABLE_CUDAGRAPH=0` → FULL_DECODE_ONLY (FULL graphs only, eager prefill); this is
  likely also what makes the classic runner correct, untested.
Not determined: mnnvl kernel argument layout (no flashinfer source locally); whether any other eager break or UVA
pointer is hit at runtime (static reading). Stage-2 reading was done by a sub-agent over the extracted sources;
the key claims above (V2 FULL capture stanza, single eager break, FI all-reduce default) were spot-checked.

## Open issues
1. Generic (non-profile) GMS reader does not work on 0.31 for GLM (third meta `.item()`, NVFP4 MoE). Use the profile.
2. The 0.31 profile is pinned to GB200 TP4/EP4, linear placement, no EPLB, vLLM 0.31.0 + transformers 5.14.1 + the
   source hashes of the 9 listed vLLM modules; any image change needs requalification.
3. Long-prefill logprobs are not bit-reproducible within one engine (Δ up to 0.49 nats, same tokens); not GMS-related.
4. vLLM's free-memory check passes for a real-KV shadow only because `patch_memory_snapshot` adds committed GMS bytes;
   the log line "Available KV cache memory: 138.65 GiB" is not meaningful (the KV size is fixed by
   `--num-gpu-blocks-override 5641` = 361,024 tokens).
5. The shadow still pays ~150 s of engine init (profiling + two graph captures) after the 6–8 s attach; that is what
   Foundry LOAD replaces (stage 2).
6. Not tested: the shadow through `python3 -m dynamo.vllm` with `DYN_VLLM_GMS_SHADOW_MODE` + failover lock (the
   offline `LLM.sleep/wake_up` path exercised the same GMSWorker sleep/wake code), and the 0.27.1 profile on its
   0.27.1 image after this refactor (unit tests only).
7. KV cache dtype on 0.31 is `fp8_e4m3` with standard MLA layout ("set `--attention-backend FLASHMLA_SPARSE` for
   fp8_ds_mla"), same for writer, shadow and no-GMS.

## Reproduce

```bash
# worktree + scripts
git -C ~/dynamo-gms-v0-glm worktree add ../dynamo-gms-v031 rvinodkumar/gms-v0-vllm031
S=~/dynamo-gms-v031/v031-scripts; NS="-n <namespace>"; P=rp-v031-a
kubectl $NS apply -f $S/pod.yaml            # 4 GPUs on w0e-...-46k1, model PVC read-only at /model-cache
# patched GMS on PYTHONPATH (+ the image's compiled allocator extension)
(cd ~/dynamo-gms-v031/lib && tar czf /tmp/gms.tgz --exclude=__pycache__ --exclude=tests gpu_memory_service)
kubectl $NS cp /tmp/gms.tgz $P:/tmp/gms.tgz
kubectl $NS exec $P -- bash -c 'mkdir -p /home/dynamo/gms-v031 /results/scripts && cd /home/dynamo/gms-v031 && tar xzf /tmp/gms.tgz && cp /usr/local/lib/python3.12/dist-packages/gpu_memory_service/client/torch/extensions/_allocator_ext*.so gpu_memory_service/client/torch/extensions/'
for f in $S/*.sh $S/*.py; do kubectl $NS cp $f $P:/results/scripts/$(basename $f); done
for f in $S/prompts*.json; do kubectl $NS cp $f $P:/results/$(basename $f); done
X="kubectl $NS exec $P -- bash -c"
# no-GMS reference (sequential protocol), then stop it
$X 'bash /results/scripts/run.sh e0b-nogms -- --role nogms --prompts /results/prompts3.json'   # waits/exit on its own
# GMS server + writer (held)
$X 'bash /results/scripts/start_gms.sh; bash /results/scripts/run.sh e1-writer -- --role writer --hold'
# shadow, real KV (any time after the writer published)
$X 'bash /results/scripts/run.sh e2-reader -- --role reader --hold'
# shadow, scratch KV: park, kill writer, wake, serve
$X 'bash /results/scripts/run.sh e3-shadow DYN_GMS_SCRATCH_KV_ENABLED=1 -- --role reader --hold --no-first-gen'
$X 'S=/results/scripts; bash $S/cmd.sh e3-shadow sleep; bash $S/kill_engines.sh e1-writer; bash $S/cmd.sh e3-shadow wake; bash $S/cmd.sh e3-shadow "gen afterwake"'
$X 'cd /results/out; python3 /results/scripts/compare.py e0b-nogms-first.json e1-writer-first.json e2-reader-first.json e3-shadow-afterwake.json'
# classic-runner diagnosis
$X 'bash /results/scripts/run.sh e10-classic VLLM_USE_V2_MODEL_RUNNER=0 -- --role reader --prompts /results/prompts3.json'
$X "bash /results/scripts/run.sh e12-classic-cap8 VLLM_USE_V2_MODEL_RUNNER=0 -- --role reader --prompts /results/prompts3.json --extra '{\"compilation_config\": {\"cudagraph_capture_sizes\": [1, 2, 4, 8]}}'"
# unit tests in the image
$X 'cd <tests dir> && PYTHONPATH=/home/dynamo/gms-v031 python3 -m pytest -q test_gms_reconstruction.py test_component_reconstruction.py'
kubectl $NS delete pod $P
```
Engine args (all engines, `engine.py`): the Snapshot-arm args (`--tensor-parallel-size 4 --enable-expert-parallel
--max-model-len 40960 --gpu-memory-utilization 0.8 --num-gpu-blocks-override 5641 --block-size 64
--max-num-batched-tokens 8192 --max-num-seqs 1024 --enable-prefix-caching --enable-chunked-prefill --seed 0
--trust-remote-code`) + `enable_flashinfer_autotune=False`; GMS engines add `load_format="gms"`,
`worker_cls=gpu_memory_service.integrations.vllm.worker.GMSWorker`,
`model_loader_extra_config={"gms_reconstruction_profile": "glm52-nvfp4-vllm031-gb200-tp4-ep4"}` (+ `"gms_read_only": true`
for shadows). `kill_engines.sh TAG` kills one engine's process group (`all` kills every engine, not the GMS server).
