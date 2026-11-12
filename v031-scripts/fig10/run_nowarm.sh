#!/bin/bash
# run_nowarm.sh: lean Foundry with skip_jit_warmup at 83 sizes: setup, primary probe, 1 baseline cycle,
# 3-cycle chain, then probe + burst on the final promoted engine.
source /results/f10/f10lib.sh
export FOUNDRY_TOML_EXTRA=${FOUNDRY_TOML_EXTRA:-'skip_jit_warmup = true'}
bash $F/setup031.sh || exit 1
cat $F/foundry-load.toml
sed 's/model="m"/model="glm"/' /results/probe.py > $F/probe_glm.py
python3 -u $F/probe_glm.py --label primary --out $F/probe-primary.json --url http://127.0.0.1:8000 > $F/logs/probe-primary.out 2>&1; tail -2 $F/logs/probe-primary.out
[ -z "$NO_BASE" ] && SEED=20261107 bash $F/arm031.sh base-c1 none 0.2
bash $F/chain031.sh foundry 3
P=$(cat /tmp/f10_target); echo "final serving port $P"
python3 -u $F/probe_glm.py --label promoted --out $F/probe-promoted.json --url http://127.0.0.1:$P > $F/logs/probe-promoted.out 2>&1; tail -2 $F/logs/probe-promoted.out
python3 /results/compare.py $F/probe-primary.json $F/probe-promoted.json 2>&1 | tail -20
python3 -u $F/burst.py $P
grep -h "Workspace is locked\|AssertionError" $F/arms/*/shadow.log $F/arms/*init-shadow.log 2>/dev/null | head -3
grep -hc "skipped warmup_kernels" $F/arms/foundry-c*/shadow.log
echo RUN_NOWARM_DONE
