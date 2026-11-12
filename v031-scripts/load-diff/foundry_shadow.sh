#!/bin/bash
# foundry_shadow.sh NAME MODEL_DIR "engine args" "base LLM kwargs json body" SIZES REGION
# GMS writer (disk) -> Foundry SAVE shadow (load_diff, fixed VA, scratch KV) -> exit
# -> Foundry LOAD shadow -> park -> kill writer -> wake -> generate; compare with the writer.
source /results/scripts/env.sh
export PYTHONPATH=/results/vllm-site:$PYTHONPATH PATH=$HOME/.local/bin:$PATH VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-1} VLLM_ALLREDUCE_USE_FLASHINFER=${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
N=$1; M=$2; A=$3; BASE=$4; SIZES=$5; REGION=${6:-256GB}; S=/results/scripts; W=/results/ldf/$N
# CG_MODE=FULL_DECODE_ONLY (with VLLM_USE_BREAKABLE_CUDAGRAPH=0) for models whose breakable prefill graphs are broken
CC="\"cudagraph_mode\": \"${CG_MODE:-FULL_AND_PIECEWISE}\", \"cudagraph_capture_sizes\": $SIZES"
mkdir -p $W; rm -rf $W/ws; mkdir -p $W/ws
for m in save load; do
  printf 'mode = "%s"\nworkspace_root = "%s"\nbase_addr = "0x600000000000"\nregion_size = "%s"\nscratch_space_size = "2GB"\n' $m $W/ws $REGION > $W/foundry-$m.toml
  # Extra TOML keys for both SAVE and LOAD, e.g. FOUNDRY_TOML_EXTRA=$'skip_capture_warmup = true\n'
  [ -n "$FOUNDRY_TOML_EXTRA" ] && printf '%s\n' "$FOUNDRY_TOML_EXTRA" >> $W/foundry-$m.toml
done
wait_for() { for i in $(seq 600); do grep -q "HOLD waiting\|SHUTDOWN_DONE\|Traceback\|Segfault" /results/logs/$1.log 2>/dev/null && break; sleep 2; done
  grep -q "HOLD waiting" /results/logs/$1.log || { echo "FAILED $1"; grep -E "Error|error:" /results/logs/$1.log | grep -v "^\s*$" | tail -6 | cut -c1-400; exit 1; }; }
SH="--role reader --hold --no-first-gen --mode load_diff --profile none --model $M $A"
bash $S/run.sh $N-writer -- --role writer --hold --mode load_diff --profile none --model $M $A \
  --extra "{$BASE, \"compilation_config\": {$CC, \"mode\": 0}}"
wait_for $N-writer
bash $S/cmd.sh $N-writer "gen second"  # the writer's own repeat noise
bash $S/run.sh $N-save DYN_GMS_FIXED_VA=1 DYN_GMS_SCRATCH_KV_ENABLED=1 -- $SH \
  --extra "{$BASE, \"compilation_config\": {$CC, \"graph_extension_config_path\": \"$W/foundry-save.toml\"}}"
wait_for $N-save
bash $S/cmd.sh $N-save exit; sleep 5; bash $S/kill_engines.sh $N-save
bash $S/run.sh $N-load DYN_GMS_FIXED_VA=1 DYN_GMS_SCRATCH_KV_ENABLED=1 -- $SH \
  --extra "{$BASE, \"compilation_config\": {$CC, \"graph_extension_config_path\": \"$W/foundry-load.toml\"}}"
wait_for $N-load
bash $S/cmd.sh $N-load sleep; bash $S/kill_engines.sh $N-writer; bash $S/cmd.sh $N-load wake
bash $S/cmd.sh $N-load "gen afterwake"; bash $S/cmd.sh $N-load "gen afterwake2"
for t in save load; do L=/results/logs/$N-$t.log
  grep -hE "TP0|EngineCore" $L | grep -oE "Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds|Graph capturing finished in [0-9]+ secs|lean: skipped profile_cudagraph_memory" | sed "s/^/  $t TP0: /"
  grep -hoE "init engine \(profile, create kv cache, warmup model\) took [0-9.]+ s" $L | sed "s/^/  $t: /"
  echo "$t: $(grep -m1 -o 'LLM_READY construct_s=[0-9.]*' $L) breakable_saved=$(grep -c 'breakable SAVE' $L) breakable_loaded=$(grep -c 'breakable LOAD' $L) restored=$(grep -c 'restored graph descriptor' $L) "
  grep -hm1 -oE '\[GMS\] load diff: applied.*owners [0-9]+' $L
  grep -hoE "\[GMS\] Reserved [0-9]+ MiB VA at 0x[0-9a-f]+" $L | head -2
done
cd /results/out && python3 $S/compare.py $N-writer-first.json $N-writer-second.json $N-load-afterwake.json $N-load-afterwake2.json
