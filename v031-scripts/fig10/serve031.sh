#!/bin/bash
# serve031.sh ROLE PORT LOG [FOUNDRY_TOML]: one `vllm serve` GLM-5.2 TP4+EP engine with the foundry_shadow.sh
# engine config (FULL_AND_PIECEWISE, 10 sizes 1..512, NCCL all-reduce, custom AR off, 5641 KV blocks).
#   ROLE=primary: GMS writer (load_diff). ROLE=shadow: GMS RO load_diff shadow, fixed VA, scratch KV.
#   FOUNDRY_TOML set: Foundry SAVE/LOAD (adapter pins torch.compile off); unset: native capture, mode 0.
# Every output line is prefixed with an epoch-ms timestamp. Process-group id -> $LOG.pgid
source /results/scripts/env.sh
export PYTHONPATH=/results/vllm-site:$PYTHONPATH PATH=$HOME/.local/bin:$PATH VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_SERVER_DEV_MODE=1
ROLE=$1; PORT=$2; LOG=$3; TOML=$4
M=/model-cache/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
# Capture sizes: $F10_SIZES (JSON list) or /results/f10/sizes.txt if present, else the 10-size bring-up list.
SIZES=${F10_SIZES:-$(cat /results/f10/sizes.txt 2>/dev/null)}; SIZES=${SIZES:-[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}
CC="\"cudagraph_mode\": \"FULL_AND_PIECEWISE\", \"cudagraph_capture_sizes\": $SIZES"
if [ -n "$TOML" ]; then CC="$CC, \"graph_extension_config_path\": \"$TOML\""; else CC="$CC, \"mode\": 0"; fi
if [ "$ROLE" = shadow ]; then
  EXTRA='{"gms_reconstruction_mode": "load_diff", "gms_read_only": true}'; ENVS="DYN_GMS_FIXED_VA=1 DYN_GMS_SCRATCH_KV_ENABLED=1"
else
  EXTRA='{"gms_reconstruction_mode": "load_diff"}'; ENVS=""
fi
setsid bash -c "env $ENVS vllm serve $M --served-model-name glm --port $PORT --trust-remote-code --seed 0 \
  --tensor-parallel-size 4 --enable-expert-parallel --max-model-len 40960 --gpu-memory-utilization 0.8 \
  --num-gpu-blocks-override 5641 --block-size 64 --max-num-batched-tokens 8192 --max-num-seqs 1024 \
  --enable-prefix-caching --enable-chunked-prefill --disable-custom-all-reduce \
  --kernel-config '{\"enable_flashinfer_autotune\": false}' \
  --load-format gms --worker-cls gpu_memory_service.integrations.vllm.worker.GMSWorker \
  --model-loader-extra-config '$EXTRA' --compilation-config '{$CC}' 2>&1 < /dev/null | python3 -u /results/f10/ts.py > $LOG" > /dev/null 2>&1 < /dev/null &
echo $! > $LOG.pgid
