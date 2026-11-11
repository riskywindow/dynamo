#!/bin/bash
# native_shadow.sh NAME MODEL_DIR "engine args" "base LLM kwargs json body" SIZES
# Baseline for foundry_shadow.sh: identical engine config (FULL_AND_PIECEWISE, torch.compile off, same sizes),
# no Foundry. GMS writer (disk) -> shadow (load_diff, fixed VA, scratch KV, native capture)
# -> park -> kill writer -> wake -> generate; compare with the writer.
source /results/scripts/env.sh
export PYTHONPATH=/results/vllm-site:$PYTHONPATH PATH=$HOME/.local/bin:$PATH VLLM_USE_BREAKABLE_CUDAGRAPH=1 VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
N=$1; M=$2; A=$3; BASE=$4; SIZES=$5; S=/results/scripts
CC="\"cudagraph_mode\": \"FULL_AND_PIECEWISE\", \"cudagraph_capture_sizes\": $SIZES, \"mode\": 0"
wait_for() { for i in $(seq 600); do grep -q "HOLD waiting\|SHUTDOWN_DONE\|Traceback\|Segfault" /results/logs/$1.log 2>/dev/null && break; sleep 2; done
  grep -q "HOLD waiting" /results/logs/$1.log || { echo "FAILED $1"; grep -E "Error|error:" /results/logs/$1.log | tail -6 | cut -c1-400; exit 1; }; }
bash $S/run.sh $N-writer -- --role writer --hold --mode load_diff --profile none --model $M $A --extra "{$BASE, \"compilation_config\": {$CC}}"
wait_for $N-writer
bash $S/run.sh $N-native DYN_GMS_FIXED_VA=1 DYN_GMS_SCRATCH_KV_ENABLED=1 -- --role reader --hold --no-first-gen --mode load_diff \
  --profile none --model $M $A --extra "{$BASE, \"compilation_config\": {$CC}}"
wait_for $N-native
bash $S/cmd.sh $N-native sleep; bash $S/kill_engines.sh $N-writer; bash $S/cmd.sh $N-native wake
bash $S/cmd.sh $N-native "gen afterwake"; bash $S/cmd.sh $N-native "gen afterwake2"
L=/results/logs/$N-native.log
echo "native: $(grep -m1 -o 'LLM_READY construct_s=[0-9.]*' $L)"
grep -h "TP0\|EngineCore" $L | grep -hoE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds|init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s|Graph capturing finished in [0-9.]+ secs" | head -6
grep -hE "CMD_DONE (sleep|wake)" $L | cut -c1-120
cd /results/out && python3 $S/compare.py $N-writer-first.json $N-native-afterwake.json $N-native-afterwake2.json
