# Foundry on vLLM 0.31: decode-only graphs (step 1 of the port)

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

## Next steps

1. Prefill graphs: breakable PIECEWISE capture (`BreakableCUDAGraphWrapper`, 79 graph segments + 78 eager closures per
   size on GLM). Hook segment begin/end with Foundry capture/save and archive the eager closures; Foundry's SGLang
   breakable-graph integration (`integration/sglang/bcg_ops.py`) is the model. This is where GLM's capture time is.
2. Collectives: restore FlashInfer's fused all-reduce (Lamport/mnnvl fabric buffers) or keep NCCL and measure its
   decode cost against the FlashInfer path.
3. Skip the memory-profiling capture on LOAD (reuse the SAVE's KV budget), then join with the GMS shadow attach
   (`lib/gpu_memory_service`, glm52-nvfp4-vllm031 profile) for a full 0.31 replenish.
