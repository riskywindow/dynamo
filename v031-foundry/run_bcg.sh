#!/bin/bash
# In-pod: breakable prefill graphs + FULL decode graphs: native reference, SAVE, LOAD, compare.  run_bcg.sh <qwenbcg|glmbcg>
PR=${1:-qwenbcg}
cd /results && mkdir -p logs out
for arm in native save load; do bash run_arm.sh $PR $arm > /results/logs/run-$PR-$arm.out 2>&1; tail -4 /results/logs/run-$PR-$arm.out; done
source /results/env.sh
python3 compare.py out/$PR-native.json out/$PR-save.json out/$PR-load.json
echo BCG_DONE $PR
