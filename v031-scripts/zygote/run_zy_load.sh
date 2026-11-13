#!/bin/bash
# run_zy_load.sh MODE: replenishment under load, lean Foundry LOAD shadow started cold after each failure (no spare),
# 83 sizes, all four matched lean switches + bytecode cache. MODE=fork (F10_FORK + cudaclean) or zygote (setup and
# the matched SAVE forked as in fork; every LOAD shadow forked from the zygote).
# Steps: setup031.sh (unless NO_SETUP), zygote (MODE=zygote), primary probe, 1 no-failure baseline cycle (unless
# NO_BASE), chain031.sh foundry NCYC (default 3; seeds 20261107 + 101*(k-1)), probe of the final promoted engine.
source /results/f10/f10lib.sh
MODE=$1
export F10_PYCACHE=${F10_PYCACHE:-/results/pycache}
export FOUNDRY_TOML_EXTRA=${FOUNDRY_TOML_EXTRA:-$'skip_jit_warmup = true\nskip_profile_run = true'}
export F10_FORK=1 RP_CUDA_CLEAN_IMPORTS=1 PYTHONPATH=/results/zy/audit${PYTHONPATH:+:$PYTHONPATH}
[ -z "$NO_SETUP" ] && { bash $F/setup031.sh || exit 1; }
if [ "$MODE" = zygote ]; then
  bash /results/zy/zystart.sh /tmp/zygote.sock $F/logs/zygote-load.log || exit 1
  export F10_ZYGOTE=/tmp/zygote.sock
fi
sed 's/model="m"/model="glm"/' /results/probe.py > $F/probe_glm.py
python3 -u $F/probe_glm.py --label primary --out $F/probe-primary.json --url http://127.0.0.1:8000 > $F/logs/probe-primary.out 2>&1; tail -1 $F/logs/probe-primary.out
[ -z "$NO_BASE" ] && SEED=20261107 bash $F/arm031.sh base-c1 none 0.2
bash $F/chain031.sh foundry ${NCYC:-3}
P=$(cat /tmp/f10_target); echo "final serving port $P"
python3 -u $F/probe_glm.py --label promoted --out $F/probe-promoted.json --url http://127.0.0.1:$P > $F/logs/probe-promoted.out 2>&1; tail -1 $F/logs/probe-promoted.out
python3 /results/compare.py $F/probe-primary.json $F/probe-promoted.json 2>&1 | tail -20
for d in $F/arms/foundry-c*; do echo "$(basename $d): $(python3 $F/startup_phases.py $d/shadow.log 2>/dev/null | sed -n 2p | cut -c41-)"; done
echo RUN_ZY_LOAD_DONE $MODE
