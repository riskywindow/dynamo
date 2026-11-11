#!/bin/bash
# pair.sh NAME MODEL_DIR "engine args" [GPUS] ["extra reader args"]: load_diff writer (held) then reader; compare outputs
source /results/scripts/env.sh
N=$1; M=$2; A=$3; G=${4:-0,1,2,3}; RA=$5; S=/results/scripts
bash $S/run.sh $N-writer CUDA_VISIBLE_DEVICES=$G -- --role writer --hold --mode load_diff --profile none --model $M $A
for i in $(seq 400); do grep -q "HOLD waiting\|Traceback" /results/logs/$N-writer.log && break; sleep 3; done
grep -E "load diff published|GEN first|Error" /results/logs/$N-writer.log | cut -c1-300 | tail -4
grep -q "HOLD waiting" /results/logs/$N-writer.log || exit 1
bash $S/run.sh $N-reader CUDA_VISIBLE_DEVICES=$G -- --role reader --mode load_diff --profile none --model $M $A $RA
for i in $(seq 400); do grep -q "SHUTDOWN_DONE\|Traceback" /results/logs/$N-reader.log && break; sleep 3; done
grep -E "\[GMS\] load diff|GEN first|Error" /results/logs/$N-reader.log | grep -v "^\s*$" | cut -c1-900 | tail -10
cd /results/out && python3 $S/compare.py $N-writer-first.json $N-reader-first.json
