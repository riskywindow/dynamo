#!/bin/bash
# In-pod (rp-v031f-*): one engine on vLLM 0.31 + the rebased Foundry adapter, FULL decode graphs only.
#   run_arm.sh <profile: qwen|glm> <arm: native|save|load> [label]
# native = same config without Foundry (reference); save = capture + archive to /results/ws-<profile>;
# load = restore the archive instead of capturing. After READY: probe.py (greedy, logprobs) -> /results/out/<label>.json
# Logs: /results/logs/<label>.log. Startup time = launch -> /v1/models answers. Stops the engine at the end.
PROFILE=$1; ARM=$2; LABEL=${3:-$PROFILE-$ARM}
source /results/env.sh
mkdir -p /results/logs /results/out
WS=/results/ws-$PROFILE; CFG=/results/foundry-$PROFILE-$ARM.toml
export VLLM_USE_BREAKABLE_CUDAGRAPH=0 HF_HUB_OFFLINE=1
case $PROFILE in
  qwen)
    M=$(ls -d /model-cache/hub/models--Qwen--Qwen3-0.6B/snapshots/* | head -1)
    SIZES='[1, 2]'; MODEL_ARGS="--dtype bfloat16 --max-model-len 128 --max-num-seqs 2 --gpu-memory-utilization 0.5"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    REGION=256GB ;;
  glm)
    M=/model-cache/hub/models--nvidia--GLM-5.2-NVFP4/snapshots/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
    SIZES=${GLM_SIZES:-'[1, 2, 4, 8, 16, 32, 64]'}
    # Snapshot-arm engine args, decode-only graphs; collectives = NCCL (PR #66472's tested TP setup)
    MODEL_ARGS="--tensor-parallel-size 4 --enable-expert-parallel --max-model-len 40960 --gpu-memory-utilization 0.8 \
      --num-gpu-blocks-override 5641 --block-size 64 --max-num-batched-tokens 8192 --max-num-seqs 1024 \
      --enable-prefix-caching --enable-chunked-prefill --disable-custom-all-reduce"
    export VLLM_ALLREDUCE_USE_FLASHINFER=0
    REGION=${GLM_REGION:-512GB} ;;
  *) echo "unknown profile $PROFILE"; exit 2 ;;
esac
CC="{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": $SIZES"
if [ "$ARM" = native ]; then
  CC="$CC, \"mode\": 0}"
else
  [ "$ARM" = save ] && { [ -e $WS ] && mv $WS $WS.prev-$(date +%s); mkdir -p $WS; }
  printf 'mode = "%s"\nworkspace_root = "%s"\nbase_addr = "0x600000000000"\nregion_size = "%s"\nscratch_space_size = "2GB"\n' \
    $ARM $WS $REGION > $CFG
  CC="$CC, \"graph_extension_config_path\": \"$CFG\"}"
fi
pkill -9 -f '[v]llm serve'; sleep 3
t0=$(date +%s.%N)
setsid nohup vllm serve $M --served-model-name m --trust-remote-code --seed 0 --disable-flashinfer-autotune \
  $MODEL_ARGS --compilation-config "$CC" --port 8000 > /results/logs/$LABEL.log 2>&1 < /dev/null &
until curl -sf localhost:8000/v1/models > /dev/null; do
  sleep 2
  pgrep -f '[v]llm serve' > /dev/null || { echo "ENGINE_DIED $LABEL"; grep -E "Error|error|Traceback" /results/logs/$LABEL.log | tail -15; exit 1; }
done
t1=$(date +%s.%N)
echo "READY $LABEL startup_s=$(python3 -c "print(round($t1-$t0,1))")"
python3 -u /results/probe.py --label $LABEL --out /results/out/$LABEL.json
echo "graphs: restored=$(grep -c 'restored graph descriptor' /results/logs/$LABEL.log) foundry_lines=$(grep -c '\[foundry\]' /results/logs/$LABEL.log)"
pkill -f '[v]llm serve'; sleep 5; pkill -9 -f '[v]llm serve'
echo "DONE $LABEL"
