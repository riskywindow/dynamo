# Model compatibility: GMS load-diff shadow + fixed VA + lean Foundry LOAD (vLLM 0.31)

2026-11-12, 4x GB200 (node o7v-9jvj, pod rp-ldf-m1), image vllm-runtime:1.6.0 + the Foundry overlay.

Flow per model (`v031-scripts/load-diff/model_run.sh` -> `foundry_shadow.sh`, lean switches
`skip_cudagraph_memory_profile` + `skip_capture_warmup`):
1. The GMS writer loads from disk and generates twice; the second generation is the writer's own repeat-noise floor.
2. A SAVE shadow (load_diff, `DYN_GMS_FIXED_VA=1`, scratch KV, Foundry SAVE) runs, then exits.
3. A LOAD shadow runs, then parks (sleep).
4. The writer is killed, the LOAD shadow wakes and generates twice.
5. Its outputs are compared with the writer's.

Engine config is the same for all models: graph mode `FULL_AND_PIECEWISE` with breakable prefill graphs,
torch.compile off, capture sizes 1..512 (10 sizes), NCCL all-reduce, custom AR off, `max_model_len` 40960,
`num_gpu_blocks_override` 4096. Prompts are `prompts.json` (6 prompts, up to ~37K tokens).

| Model | Arch / quant | TP/EP | Load diff (reader) | Foundry LOAD | Post-wake vs writer |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite (email: "crashes") | DeepseekV2 MLA MoE, BF16 | TP1 | 54 CPU tensors restored, 0 missing (after the identity fix below) | 10 breakable + 9 decode restored | 6/6 identical, Δ 0 (both generations) |
| Qwen3-30B-A3B-FP8 | Qwen3 MoE, block FP8 [128,128] | TP4+EP | 96 objects linked (`deep_gemm_warmup_provider`), 0 missing | 40 breakable + 36 decode | 6/6 identical, Δ 0 |
| Qwen3.6-35B-A3B-FP8 (text-only) | Qwen3.5-MoE, hybrid linear attention, block FP8 [128,128] | TP4+EP | 0 missing (20 KV-scale placement warnings) | 40 + 36 | 6/6 identical, Δ 0 |
| Nemotron-3.5-Lightning-30B-A3B | NemotronH hybrid Mamba MoE, BF16 | TP4+EP | 12 CPU tensors restored, 0 missing | decode-only: 36 restored (see note) | 6/6 identical, Δ 0 |
| GLM-5.3-Flash-NVFP4 (text-only) | Glm5Next DSA, ModelOpt NVFP4 | TP4+EP | 6 values applied (`is_w4a16_nvfp4`), 22 CPU tensors, 0 missing | 40 + 36 | short 4/4 identical Δ 0; long prompts diverge at tokens 14/–, within the writer's own repeat noise (15/5, Δ ≤ 0.73) |
| gpt-oss-120b | GptOss MoE, MXFP4 | TP4+EP | **refused at attach**: `GptOssMxfp4MoEMethod` has no pre-processed-weights path | – | – |
| GLM-5.2-FP8 (email: "refuses") | GlmMoeDsa, block FP8 | – | not run: 756 GB > ~737 GB HBM of one 4x GB200 node, and not cached | – | – |

GLM-5.2-FP8 uses the same block-scaled FP8 format (e4m3, `weight_block_size` [128, 128], dynamic activations)
as Qwen3-30B-A3B-FP8 and Qwen3.6-35B-A3B-FP8. Both of those pass, so the FP8 load-time processing that
Zane's weight cache refused (block-scale repack, DeepGEMM layout) is handled. The model itself needs 8 GPUs.

## Failures and root causes

1. **DeepSeek-V2-Lite: identity mismatch (`config_sha256`). Fixed in load_diff, generic.**
   - vLLM's DeepSeek-V2 model mutates the HF config during construction/loading.
   - The writer fingerprinted at publish time; readers fingerprint before constructing.
   - The writer now takes its identity before `initialize_model` (`model_loader.py`, `load_diff.publish_descriptor`). There is a unit test.
2. **Nemotron-H: garbage output (`\x17…`, NaN logprobs) on short prompts. Engine-level, not GMS.**
   - It reproduces without GMS under vLLM 0.31's breakable `FULL_AND_PIECEWISE` graphs.
   - Eager mode and `FULL_DECODE_ONLY` without breakable graphs are correct.
   - So the shadow path was run with `CG_MODE=FULL_DECODE_ONLY VLLM_USE_BREAKABLE_CUDAGRAPH=0`, which passes.
   - This is a vLLM bug in breakable graphs on hybrid Mamba layers.
3. **gpt-oss-120b: refused at attach. vLLM gap; a per-quant-method change, not per model.**
   - The generic `Mxfp4MoEMethod` in 0.31 has a `weights_already_processed` branch that only rebuilds the kernel (`_build_moe_kernel`).
   - `GptOssMxfp4MoEMethod.process_weights_after_loading` always repacks through `_setup_kernel`, and it doesn't declare `supports_pre_processed_weights`.
   - The same branch is needed there. vLLM's own weight-cache IPC loader refuses it for the same reason.

Benign warnings seen:
- `_k_scale_cpu`/`_v_scale_cpu`: CUDA in the writer vs CPU after the reader's KV-cache rebuild, value 1.0 on both sides.
- DeepSeek-V2-Lite: one container created at load (`_pending_indexer_wk_fp8`, a dict).

Results: `~/load-diff-20261111/results-models/`.
