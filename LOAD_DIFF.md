# Load diff: model-agnostic GMS read-only attach (vLLM 0.31)

2026-11-11. Idea from Zane Sun's SGLang weight-cache work ("weight cache for models with
load-time processing: load diff"); built independently on our GMS v0 reader.

## What it is

`gms_reconstruction_mode="load_diff"` (writer and readers) in `model_loader_extra_config`. No model, GPU,
parallelism or vLLM version is pinned; the GLM profile path (`glm_reconstruction.py`) is untouched.

- **Writer** (`_load_write_mode`): snapshots every module, its `quant_method` and attention `impl` right after
  `initialize_model` and again after `load_weights` + `process_weights_after_loading`; publishes
  `__gms__.load_diff.v1` = final state + which attributes loading changed + module schema/aliases/expert-map
  digests + a generic fingerprint (config, versions, parallelism, ranks, device, hashes of the source files of
  every class in the model).
- **Reader** (`load_diff.restore`):
  1. meta construction (0.31 expert-map `.item()` workaround kept, guarded), drop params loading deleted,
     attach GMS tensors;
  2. rebuild Python objects: each quant method's `process_weights_after_loading` under vLLM's
     `weights_already_processed()` (non-quant post-processing is skipped; its outputs are attached tensors and
     the mapping is read-only);
  3. apply the diff: simple values loading set come from the writer; attributes loading deleted are deleted;
     small CPU tensors travel by value (restored when the reader's are left on meta); objects loading wired
     into a new attribute are re-linked by reference (`ref`);
  4. fatal only if an object/tensor loading created is still missing or a tensor is left on meta; value and
     placement disagreements are logged as warnings.

Code: `lib/gpu_memory_service/integrations/vllm/load_diff.py` (+ `model_loader.py` wiring,
`component_reconstruction.py` mode name). Tests: `tests/test_load_diff.py` (11 CPU tests).

## Results (4x GB200, vLLM 0.31.0, image vllm-runtime:1.6.0, pod rp-ldiff-a on w0e-46k1)

Writer (disk load, held) then load-diff reader on the same GPUs; greedy outputs of `prompts.json`
(6 prompts, up to 30K tokens) compared with `compare.py`.

| Model | Config | Changed by loading | Reader result |
|---|---|---|---|
| Qwen3-0.6B | TP1 | 56 attrs | identical to writer, Δlogprob 0 (2 prompts); 56 CPU KV-scale tensors restored by value |
| GLM-5.2-NVFP4 | TP4+EP4 | 150 attrs / 75 owners | 6/6 identical; Δ 0 short, 0.43/0.40 on 8K/30K = known long-prefill noise (NOTES.md open issue 3); 0 values needed applying (0.31's pre-processed rebuild reproduces all of them, i.e. the profile's SCALAR_FIELDS copy is redundant on 0.31) |
| Qwen3-30B-A3B-FP8 | TP4+EP4 | 384 attrs / 240 owners | bit-identical (Δ 0, 6/6) to an independent disk-loaded engine (q30d writer); its own writer (q30e) differs from q30d's writer by the same amount, i.e. run-to-run variation of this FP8 stack, not GMS; 96 `deep_gemm_warmup_provider` objects re-linked |
| GLM-5.3-Flash-NVFP4 (`Glm5NextForConditionalGeneration`) | TP4+EP4, text-only | 112 attrs / 59 owners | 4/4 short prompts identical, Δ 0; 8K/30K diverge at tokens 5/— (Δ 0.003/0.76), while the writer repeating the same prompts diverges from itself at tokens 5/6 (Δ 0.006/0.21), i.e. long-prefill noise; 6 values applied that vLLM's rebuild does not reproduce (`is_w4a16_nvfp4`), 22 CPU tensors restored |

GLM-5.2 regression with the final code (run glm52f): 4/4 short prompts identical, Δ 0; long30k same tokens
(Δ 0.12); long8k same answer, then the writer emits EOS at token 6 where the reader continues (Δ 0.03). The
writer repeating the same prompts varies by Δ 0.14-0.47 on these two prompts (same tokens in 3 samples), so the
EOS flip is a near-tie inside that noise, but it is the only token difference seen on GLM-5.2.

Raw logs and outputs: `~/load-diff-20261111/results/{logs,out}` (pod deleted).

Warnings seen (not fatal): `_k_scale_cpu`/`_v_scale_cpu` live on CUDA in the writer (constructed under the
device context) but on CPU after the reader's KV-cache rebuild; value 1.0 on both.

## Replenishment flow with a load-diff shadow (GLM-5.2-NVFP4 TP4/EP4)

`shadow.sh`: writer (disk load, held) -> shadow (`load_diff`, `DYN_GMS_SCRATCH_KV_ENABLED=1`, held) -> park
(`sleep`) -> kill writer -> `wake` -> generate twice. Both post-wake generations: 6/6 token-identical to the
writer; short prompts Δ 0, long prompts Δ 0.03-0.23 (writer's own repeat noise 0.14-0.47).

| | load diff (2026-11-11) | GLM profile path (NOTES.md) |
|---|---|---|
| weight attach (`Model loading took`) | 7.2 s, 106.33 GiB imported | 6-8 s |
| shadow construct (`LLM()`, incl. profiling + graph capture) | 158.4 s | 155 s |
| device memory, writer + scratch-KV shadow | 148,927 MiB per GPU | 148.93 (same counter) |
| park / wake | 0.22 s / 4.35 s | - / 4.3 s |

Results: `~/load-diff-20261111/results-shadow/`.

## Foundry graph SAVE/LOAD on load-diff shadows (full 0.31 replenish path)

**Problem.** Captured graphs hold raw weight and KV pointers. GMS reserved VA with hint 0, so (a) addresses differ
per process, and (b) under Foundry's `libcuda_hook` every hint-0 reserve < 2 GB is carved from Foundry's arena, which
LOAD pre-maps with fresh physical memory before weights load: GMS's `cuMemMap` would hit mapped VA (the lean-v4
INVALID_VALUE) and the archived span would include a second physical copy of the weights.

**Fix: GMS fixed-VA mode** (`DYN_GMS_FIXED_VA=1`, `client/memory_manager.py`). Each tag reserves one whole window
when its manager is created (weights `0x420000000000` + 512 GiB, KV `0x4A0000000000` + 256 GiB; override
`DYN_GMS_FIXED_VA_BASE_<TAG>` / `_SIZE_<TAG>`), and allocations are carved from it: imports at
`base + offset(layout slot)` (independent of import order and process), scratch/KV reservations by cursor. Hinted
reserves pass straight through Foundry's hook, so nothing lands in its arena. The window must be reserved early: per
allocation, the driver ignored the hint (`0x700000000000` -> `0xf4e667c00000` on the first try; peer-mapping step 1
saw 0/21 honored at LOAD); a failed window reservation is fatal with a clear message.

**Flow** (`v031-scripts/load-diff/foundry_shadow.sh`): GMS writer (disk) -> SAVE shadow (load_diff, fixed VA,
scratch KV, Foundry SAVE) -> exit -> LOAD shadow (same, Foundry LOAD) -> park -> kill writer -> wake -> generate
twice. Engine config = the Foundry session's breakable profiles (`FULL_AND_PIECEWISE`, torch.compile off, NCCL
all-reduce, custom AR off); the writer uses the same config without Foundry.

| | Qwen3-0.6B TP1 | GLM-5.2-NVFP4 TP4+EP |
|---|---|---|
| SAVE shadow | 8 breakable prefill sizes archived | 40 breakable sets (10 sizes x 4 ranks) archived |
| LOAD shadow | 8 breakable + 4 decode restored, no capture | 40 breakable + 40 decode descriptors restored, no capture |
| post-wake vs writer | 2/2 identical, Δ 0, both generations | short 4/4 identical Δ 0; long prompts Δ <= 0.45; one EOS near-tie at token 6 in 1 of 2 generations (Δ 0.013) |
| KV / weight VA, both shadows | 0x4a0000000000 / 0x420000000000 | same |
| GLM LOAD shadow (TP0) | | attach 8.1 s; init engine 20.8 s (profiling capture 8 s + restore 5 s vs SAVE capture 14 s); construct 77.2 s (SAVE 90.6 s); wake 4.25 s |

Not measured yet: a native (no-Foundry) shadow with the same engine config, for the startup comparison; under load.
Results: `~/load-diff-20261111/results-foundry-{qwen,glm}/`.

## Bugs/gaps found by the second models

1. **Constructor CPU tensors left on meta** (Qwen3-0.6B, unquantized: no KV-cache quant method rebuilds
   `_k_scale_cpu`). Fixed generically: small CPU tensors (<= 4096 elements) are carried by value.
2. **GMS v0 writer crash on FakeTensors** (Qwen3-30B-A3B-FP8): `copy_non_parameter_tensors_to_default_allocator`
   scans every live tensor (`gc.get_objects()`) and called `data_ptr()` on torch.compile FakeTensors. Pre-existing
   bug in the v0 writer, not load-diff specific. Fixed in `client/torch/storage.py::_containing_mapping`.
3. **Object loading creates that the pre-processed rebuild skips** (Qwen3-30B-A3B-FP8): vLLM's DeepGEMM FP8
   linear kernel stamps itself onto its layer as `deep_gemm_warmup_provider` in normal post-processing; the
   pre-processed path (`Fp8LinearMethod.process_weights_after_loading` early return) does not. Only DeepGEMM
   JIT warmup reads it (missing => warmup silently skips those layers; outputs unaffected). vLLM's own
   weight-cache IPC loader has the same gap upstream. Load diff now re-links it by reference.
4. **Zero-element buffers left on meta** (GLM-5.3-Flash): the DSA indexer rope has `rotary_dim=0`, so its
   `cos_sin_cache` is `[1048576, 0]`; GMS publishes nothing for it. Zero-element params/buffers are now
   materialized empty on the device.
5. **Values loading set that the pre-processed rebuild does not** (GLM-5.3-Flash): `is_w4a16_nvfp4` on 6
   MLP linears. Applied from the writer; a hand-written scalar list would have missed them.

Not GMS-related: GLM-5.3-Flash segfaults in a cuDNN conv of the vision tower during vLLM's profiling run on
this image with or without GMS; run text-only (`limit_mm_per_prompt={"image":0,"video":0}`).
Qwen3-30B-A3B-FP8 is not run-to-run deterministic on this stack (two disk-loaded engines differ on 3/6 prompts).

## Reproduce

Scripts: `~/load-diff-20261111/` (`deploy.sh`, `pair.sh NAME MODEL "engine args"`,
`engine.py --mode load_diff --profile none --model ... --tp N [--glm-args 0]`).
Kill GMS servers with `pkill -9 -f "[g]pu_memory_service --device"` (the old `cli.server` pattern does not match)
and wait for 0 MiB before the next writer; otherwise a committed layout survives and the next "writer" attaches RO.
