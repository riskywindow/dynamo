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
# Startup options: F10_PYCACHE=<dir> -> PYTHONPYCACHEPREFIX (the image's site-packages ship few .pyc and are not
# writable); F10_GATE=<file> -> shadow holds after weight attach until <file> exists (pre-started spare).
[ -n "$F10_PYCACHE" ] && export PYTHONPYCACHEPREFIX=$F10_PYCACHE
# F10_FORK=1: fork the engine core and workers instead of spawning them (skips two Python bootstraps). Foundry's hook
# must then be preloaded for the whole process tree with the env its spawn-site injection would have set.
if [ "$F10_FORK" = 1 ]; then
  export VLLM_WORKER_MULTIPROC_METHOD=fork
  if [ -n "$TOML" ]; then
    export LD_PRELOAD=/home/dynamo/.local/lib/python3.12/site-packages/foundry/libcuda_hook.so${LD_PRELOAD:+:$LD_PRELOAD}
    export FOUNDRY_MODE=$(sed -n 's/^mode = "\(.*\)"/\1/p' $TOML) FOUNDRY_EARLY_RESERVE_BASE=0x600000000000 \
      FOUNDRY_EARLY_RESERVE_SIZE=549755813888
  fi
fi
date +%s.%N > $LOG.launch
M=/model-cache/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
# Capture sizes: $F10_SIZES (JSON list) or /results/f10/sizes.txt if present, else the 10-size bring-up list.
SIZES=${F10_SIZES:-$(cat /results/f10/sizes.txt 2>/dev/null)}; SIZES=${SIZES:-[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}
CC="\"cudagraph_mode\": \"FULL_AND_PIECEWISE\", \"cudagraph_capture_sizes\": $SIZES"
if [ -n "$TOML" ]; then CC="$CC, \"graph_extension_config_path\": \"$TOML\""; else CC="$CC, \"mode\": 0"; fi
if [ "$ROLE" = shadow ]; then
  EXTRA='{"gms_reconstruction_mode": "load_diff", "gms_read_only": true}'; ENVS="DYN_GMS_FIXED_VA=1 DYN_GMS_SCRATCH_KV_ENABLED=1"
  [ -n "$F10_GATE" ] && ENVS="$ENVS DYN_GMS_STARTUP_GATE=$F10_GATE VLLM_ENGINE_READY_TIMEOUT_S=86400"
else
  EXTRA='{"gms_reconstruction_mode": "load_diff"}'; ENVS=""
fi
# F10_ZYGOTE=<sock>: a shadow is forked from the zygote (v031-scripts/zygote/zygote.py, started by zystart.sh with the
# LOAD shadow's environment) instead of being launched as a new process: same arguments, imports already done.
if [ -n "$F10_ZYGOTE" ] && [ "$ROLE" = shadow ]; then
  python3 /results/zy/zyctl.py $F10_ZYGOTE $LOG $ENVS -- serve $M --served-model-name glm --port $PORT --trust-remote-code \
    --seed 0 --tensor-parallel-size 4 --enable-expert-parallel --max-model-len 40960 --gpu-memory-utilization 0.8 \
    --num-gpu-blocks-override 5641 --block-size 64 --max-num-batched-tokens 8192 --max-num-seqs 1024 \
    --enable-prefix-caching --enable-chunked-prefill --disable-custom-all-reduce \
    --kernel-config '{"enable_flashinfer_autotune": false}' \
    --load-format gms --worker-cls gpu_memory_service.integrations.vllm.worker.GMSWorker \
    --model-loader-extra-config "$EXTRA" --compilation-config "{$CC}" > $LOG.zyctl
  exit 0
fi
setsid bash -c "env $ENVS vllm serve $M --served-model-name glm --port $PORT --trust-remote-code --seed 0 \
  --tensor-parallel-size 4 --enable-expert-parallel --max-model-len 40960 --gpu-memory-utilization 0.8 \
  --num-gpu-blocks-override 5641 --block-size 64 --max-num-batched-tokens 8192 --max-num-seqs 1024 \
  --enable-prefix-caching --enable-chunked-prefill --disable-custom-all-reduce \
  --kernel-config '{\"enable_flashinfer_autotune\": false}' \
  --load-format gms --worker-cls gpu_memory_service.integrations.vllm.worker.GMSWorker \
  --model-loader-extra-config '$EXTRA' --compilation-config '{$CC}' 2>&1 < /dev/null | python3 -u /results/f10/ts.py > $LOG" > /dev/null 2>&1 < /dev/null &
echo $! > $LOG.pgid
