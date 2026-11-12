# Foundry on vLLM 0.31: decode graphs (step 1) and breakable prefill graphs (step 2)

Status 2026-11-11: FULL decode CUDA graphs SAVE and LOAD on vLLM 0.31.0 (the Snapshot arm's image,
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0`), for Qwen3-0.6B on one GPU and for GLM-5.2 NVFP4 TP4+EP on 4x GB200.
LOAD restores the archived graphs instead of capturing them and serves bit-identical outputs to a native engine.

## What was ported

- `overlay/` = vLLM PR #66472 ("[Fast Start] Persist CUDA graphs with Foundry", vllm-project/vllm, open) applied to
  vLLM 0.31.0. That PR already targets the V2 GPU model runner and FULL decode graphs. Rebase notes:
  - `v1/worker/gpu/cudagraph_utils.py` `CudaGraphManager.capture`: hand-merged; 0.31 sets the graph pool with
    `set_graph_pool_id` + `torch.cuda.graph(graph, self.pool)` instead of main's `capture_pool(...)`.
  - The extensible-KV-cache guard in `config/vllm.py` is dropped (the feature does not exist in 0.31).
  - Not taken: the Fast Start preload / weight daemon (`entrypoints/cli/preload.py`, `weight_cache/daemon.py`).
  - `vllm031-foundry.diff` = the adapter + core changes as one diff against 0.31.0.
- Added for GLM (`overlay/vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py`): `_can_use_flashinfer` now
  honors `VLLM_ALLREDUCE_USE_FLASHINFER=0`. The DSA decoder calls FlashInfer's fused all-reduce + RMSNorm directly,
  bypassing that env; with the guard, decode graphs use the regular all-reduce (NCCL, custom AR disabled) + RMSNorm,
  the TP setup PR #66472 tested. FlashInfer Lamport/mnnvl buffers are not restorable by Foundry yet.
- Foundry: upstream `foundry-org/foundry` main 3e034b0 (0.1.0rc4) built from source in the pod (no aarch64 wheels):
  user site, `cmake>=4` from pip, CUDA dev headers missing from the image's toolkit taken from torch's pip CUDA bundle
  (`nvidia/cu13/include`, e.g. cusparse.h) and the `nvidia-nvml-dev==13.*` wheel (nvml.h). See `build_foundry.sh`.
- The image runs as a non-root user: the overlay is applied to a copy of the image's vllm package in
  `/results/vllm-site`, put first on `PYTHONPATH` by `env.sh`.

0.31 specifics: breakable CUDA graphs must be off (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`, otherwise 0.31 auto-enables them
for some architectures); `cudagraph_mode=FULL_DECODE_ONLY` with `cudagraph_capture_sizes`; FlashInfer autotune off via
`--kernel-config '{"enable_flashinfer_autotune": false}'` (no `--disable-flashinfer-autotune` flag in 0.31). The adapter
pins torch.compile off on SAVE and LOAD; the native reference uses `"mode": 0` to match.

## Results (pod rp-v031f-a on o7v-...-8wwv; `results/`)

Qwen3-0.6B, TP1, decode sizes [1, 2]:
- LOAD restored 2/2 graphs. Solo, pair and batch-8 outputs identical to native (logprob delta 0.0000). SAVE too.
- Startup launch -> ready: native 62.6 s, SAVE 52.5 s, LOAD 44.4 s.

GLM-5.2 NVFP4, TP4 + EP, Snapshot-arm engine args + `--disable-custom-all-reduce`, `VLLM_ALLREDUCE_USE_FLASHINFER=0`,
decode sizes [1, 2, 4, 8, 16, 32, 64]:
- LOAD restored 28 graphs (7 sizes x 4 ranks) in ~3 s; archive 496 MB.
- LOAD vs a warm native engine: solo 4/4, pair 4/4, batch-8 8/8 identical, logprob delta 0.0000.
- Native vs native (two engines): batch-8 2/8 identical (max delta 0.675), the same divergence as LOAD vs the first
  native engine; concurrent requests batch differently run to run. SAVE vs native: within the same noise.
- Startup (warm page cache): native 177.9 s, SAVE 179.9 s, LOAD 175.9 s. Weight loading is ~80 s of that; native
  decode-only capture of 7 sizes takes only 4 s (+ a 7 s memory-profiling capture, which LOAD still runs), so
  decode-only persistence saves little on GLM. The time is in prefill graphs (breakable capture, 59 s in stage 1).

## Step 2: breakable prefill graphs (`overlay/vllm/compilation/foundry_adapter/breakable.py`)

0.31 captures prefill (PIECEWISE) graphs as breakable captures: an ordered chain of plain graph segments with an eager
closure per attention layer (`lambda: fn(*weak_args, **weak_kwargs)` from `@eager_break_during_capture`). The module
patches `BreakableCUDAGraphWrapper._capture`:
- SAVE: each segment is captured into a Foundry graph (same pool, current stream) and saved under the adapter's filename
  scheme (mode PIECEWISE, segment index in the piecewise slot); `bcg_<descriptor>.json` records the order of segments
  and breaks, each break's callee (module + qualname), its arguments (tensors as ptr/shape/stride/dtype, modules by
  their path in the model, `LayerName` and scalars by value) and the entry's output tensors.
- LOAD: no capture. Segments come back through `finish_one_graph_load` in archived order; closures are rebuilt from the
  archive (non-owning tensor views via `__cuda_array_interface__`, address checked against the archive; modules looked up
  by path; callee re-imported). The capture loop's per-size eager warmup still runs, keeping Foundry's allocation
  trajectory equal to SAVE's.
- Profiles `qwenbcg` / `glmbcg` in `run_arm.sh` (`VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `FULL_AND_PIECEWISE`); `run_bcg.sh`.

Results (`results/qwenbcg`, `results/glmbcg`):
- Qwen3-0.6B: 8 prefill sizes x (29 segments + 28 breaks) archived; LOAD rebuilds each in ~5 ms. LOAD vs SAVE
  token-identical in every set (solo, pair, batch-8, long). SAVE and LOAD both differ from native on one pair prompt at
  token 21 (capture-path numerics, same place in both).
- GLM-5.2 TP4+EP (NCCL all-reduce): 10 prefill sizes (1..512 tokens) x 4 ranks, each 79 segments + 78 DSA-attention
  breaks (bound methods, resolved by module path) + 10 FULL decode graphs per rank; archive 1.3 GB. LOAD rebuilds a
  prefill size in ~25 ms. Outputs vs native: solo 4/4, pair 4/4, long 2/2 identical (delta 0); batch-8 6/8 (concurrent
  batching; native-vs-native was 2/8). LOAD vs SAVE: solo/pair/long identical, batch-8 6/8.
- Graph phase per engine (TP0): native profiling capture 8 s + final capture 9 s (init engine 27.4 s); SAVE 8 + 16 s
  (35.8 s); LOAD 9 s profiling capture + 6 s warmups/restore (23.8 s). Startup totals are dominated by weight loading
  from the model PVC (92-235 s, highly variable), which a GMS shadow attach replaces. With 10 sizes up to 512 tokens,
  GLM's capture on 0.31 is only ~9 s; the remaining LOAD graph time is the per-size eager warmup forwards and the
  profiling capture.

## Run

```bash
NS="-n <namespace>"; P=rp-v031f-a; K=~/dynamo-gms-v031/v031-foundry
kubectl $NS apply -f $K/pod.yaml
git -C ~/foundry-gms-capture-lifecycle archive --format=tar.gz --prefix=foundry/ -o /tmp/foundry-main.tgz origin/main
kubectl $NS cp /tmp/foundry-main.tgz $P:/results/foundry-main.tgz
kubectl $NS cp $K/overlay $P:/results/overlay
for f in build_foundry.sh env.sh run_arm.sh probe.py compare.py run_qwen.sh run_glm.sh; do kubectl $NS cp $K/$f $P:/results/$f; done
kubectl $NS exec $P -- bash /results/build_foundry.sh          # ~10 min; prints BUILD_DONE
kubectl $NS exec $P -- bash /results/run_qwen.sh               # native, save, load, compare
kubectl $NS exec $P -- bash /results/run_glm.sh                # same for GLM-5.2 TP4+EP
```

## Step 3: lean LOAD on GMS load-diff shadows (`skip_cudagraph_memory_profile`, `skip_capture_warmup`)

Two Foundry TOML keys, set identically on SAVE and LOAD so the allocation trajectories stay matched
(`FOUNDRY_TOML_EXTRA` in `v031-scripts/load-diff/foundry_shadow.sh`): `skip_cudagraph_memory_profile` makes
`profile_cudagraph_memory` return 0 (KV geometry must be fixed, here `num_gpu_blocks_override=5641` + GMS scratch KV
outside the region); `skip_capture_warmup` drops the per-size eager warmup forward in the V2 capture loop.
GLM-5.2 NVFP4 TP4+EP, GMS writer -> SAVE shadow -> LOAD shadow -> park -> kill writer -> wake (TP0):

| LOAD shadow | baseline | + skip profile capture | + skip warmups |
|---|---|---|---|
| init engine | 20.75 s | 13.28 s | 7.98 s |
| graph phase (profile capture + restore) | 8 s + 5-6 s | 6 s | 1 s |
| construct (`LLM()`) | 77.2 s | 66.9 s | 64.0 s |
| wake | 4.25 s | 4.36 s | 5.03 s |

Native shadow, same config, no Foundry (main branch 508f193438): construct 80.7 s, init engine 26.7 s, capture 9+9 s.
Correctness (both post-wake generations vs the writer): short prompts 4/4 identical, logprob delta 0; long prompts
same answer, delta <= 0.30, with the same EOS near-tie at token 6 in one generation as before the change.
SAVE with both keys: init 21.1 s, capture 11 s.

Forwards still run by a lean LOAD shadow, in order (TP0 timeline, 23:10:40 = weights attached):
1. `profile_run` (GMS scratch-KV `determine_available_memory`): one forward at `max_num_batched_tokens` = 8192 tokens,
   attention skipped (MoE + all-reduce run), + dummy sampler; ~3 s.
2. V2 `warmup_kernels` (`kernel_config.enable_jit_warmup`, default on, no log line): a prefill step of
   min(max_num_seqs, KV blocks) = 1024 requests x 2 tokens (2048 tokens), then 3 uniform decode steps of 1024 tokens
   (eager: above the 512 capture limit), + sampling; ~2 s. vLLM keeps it so workspaces are sized before capture.
3. Graph restore: no forwards (1 s). ~3 s after capture before "init engine" returns, not attributed.
Removed: the profiling capture (graphs + one warmup each) and 20 per-size warmups (10 PIECEWISE 1..512 + FULL decode).
The two remaining large collective forwards (8192-token profile, 1024-request JIT warmup) are the next interference
candidates; replacing profile_run needs the workspace sizing it provides (V2 locks workspaces after capture).

## Next steps

1. LOAD still runs the eager warmup forward per prefill size and vLLM's memory-profiling capture (~9 s each on GLM).
   Dropping them needs a matched SAVE (same allocation trajectory without the warmups), as the 0.27.1 lean-load work did,
   and reusing SAVE's KV budget instead of profiling.
2. Collectives: restore FlashInfer's fused all-reduce (Lamport/mnnvl fabric buffers) or keep NCCL and measure its
   decode cost against the FlashInfer path.
3. Skip the memory-profiling capture on LOAD (reuse the SAVE's KV budget), then join with the GMS shadow attach
   (`lib/gpu_memory_service`, glm52-nvfp4-vllm031 profile) for a full 0.31 replenish.
