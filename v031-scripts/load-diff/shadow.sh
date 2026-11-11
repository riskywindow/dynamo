#!/bin/bash
# shadow.sh NAME MODEL "engine args": GMS v0 replenishment flow with a load_diff shadow.
# writer (disk load, held, gen first) -> shadow (RO attach, scratch KV, held) -> park (sleep)
# -> kill writer -> wake -> gen afterwake; compare against the writer's outputs.
source /results/scripts/env.sh
N=$1; M=$2; A=$3; S=/results/scripts
bash $S/run.sh $N-writer -- --role writer --hold --mode load_diff --profile none --model $M $A
for i in $(seq 400); do grep -q "HOLD waiting\|Traceback" /results/logs/$N-writer.log && break; sleep 3; done
grep -q "HOLD waiting" /results/logs/$N-writer.log || { grep -E "Error" /results/logs/$N-writer.log | tail -3; exit 1; }
bash $S/run.sh $N-shadow DYN_GMS_SCRATCH_KV_ENABLED=1 -- --role reader --hold --no-first-gen --mode load_diff --profile none --model $M $A
for i in $(seq 400); do grep -q "HOLD waiting\|Traceback" /results/logs/$N-shadow.log && break; sleep 3; done
grep -hE "\[GMS\] load diff:|LLM_READY" /results/logs/$N-shadow.log | cut -c1-300 | tail -3
grep -q "HOLD waiting" /results/logs/$N-shadow.log || { grep -E "Error" /results/logs/$N-shadow.log | tail -3; exit 1; }
bash $S/cmd.sh $N-shadow "mem before-park"
bash $S/cmd.sh $N-shadow sleep
bash $S/kill_engines.sh $N-writer
bash $S/cmd.sh $N-shadow wake
bash $S/cmd.sh $N-shadow "gen afterwake"
bash $S/cmd.sh $N-shadow "gen afterwake2"
grep -hE "CMD_DONE" /results/logs/$N-shadow.log | cut -c1-200
cd /results/out && python3 $S/compare.py $N-writer-first.json $N-shadow-afterwake.json $N-shadow-afterwake2.json
