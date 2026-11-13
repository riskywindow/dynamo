#!/bin/bash
# zystart.sh SOCK LOG: start the zygote (zygote.py) with a lean Foundry LOAD shadow's environment (as serve031.sh with
# F10_FORK=1 and the load TOML would set it) and wait until it listens on SOCK. Needs /results/f10/foundry-load.toml.
source /results/scripts/env.sh
export PYTHONPATH=/results/vllm-site:/results/zy/audit:$PYTHONPATH PATH=$HOME/.local/bin:$PATH VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  VLLM_ALLREDUCE_USE_FLASHINFER=0 VLLM_SERVER_DEV_MODE=1 VLLM_WORKER_MULTIPROC_METHOD=fork RP_CUDA_CLEAN_IMPORTS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
[ -n "$F10_PYCACHE" ] && export PYTHONPYCACHEPREFIX=$F10_PYCACHE
TOML=/results/f10/foundry-load.toml
export LD_PRELOAD=/home/dynamo/.local/lib/python3.12/site-packages/foundry/libcuda_hook.so${LD_PRELOAD:+:$LD_PRELOAD}
export FOUNDRY_MODE=$(sed -n 's/^mode = "\(.*\)"/\1/p' $TOML) FOUNDRY_EARLY_RESERVE_BASE=0x600000000000 \
  FOUNDRY_EARLY_RESERVE_SIZE=549755813888 DYN_GMS_FIXED_VA=1 DYN_GMS_SCRATCH_KV_ENABLED=1
pkill -9 -f "[z]ygote.py"; rm -f $1
t=$(date +%s.%N)
setsid python3 -u /results/zy/zygote.py $1 > $2 2>&1 < /dev/null &
for i in $(seq 600); do [ -S $1 ] && grep -q "ZYGOTE.* ready on" $2 && break
  pgrep -f "[z]ygote.py" > /dev/null || { echo "ZYGOTE DIED"; tail -20 $2; exit 1; }; sleep 0.5; done
echo "zygote up in $(python3 -c "import time;print(round(time.time()-$t,1))") s"
grep -E "ZYGOTE.*(FAILED|preimport done|ready|WARNING)" $2 | cut -c1-300
[ -S $1 ]
