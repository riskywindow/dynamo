#!/bin/bash
# In-pod: decode-only GLM-5.2 NVFP4 TP4+EP (NCCL all-reduce): native reference, SAVE, LOAD, compare.
cd /results && mkdir -p logs out
for arm in native save load; do bash run_arm.sh glm $arm > /results/logs/run-glm-$arm.out 2>&1; tail -4 /results/logs/run-glm-$arm.out; done
source /results/env.sh
python3 compare.py out/glm-native.json out/glm-save.json out/glm-load.json
echo GLM_DONE
