#!/bin/bash
# In-pod: decode-only mechanics on Qwen3-0.6B, one GPU: native reference, SAVE, LOAD, compare.
cd /results && mkdir -p logs out
for arm in native save load; do bash run_arm.sh qwen $arm > /results/logs/run-qwen-$arm.out 2>&1; tail -4 /results/logs/run-qwen-$arm.out; done
source /results/env.sh
python3 compare.py out/qwen-native.json out/qwen-save.json out/qwen-load.json
echo QWEN_DONE
