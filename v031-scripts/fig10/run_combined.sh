#!/bin/bash
# run_combined.sh: everything together at the 83 default capture sizes. Gated pre-started spare (chain031g.sh),
# bytecode cache (F10_PYCACHE), and all four matched lean switches (setup031.sh writes skip_cudagraph_memory_profile +
# skip_capture_warmup; FOUNDRY_TOML_EXTRA adds skip_jit_warmup + skip_profile_run).
# Steps: setup (GMS, primary, matched SAVE), primary probe, 1 no-failure baseline cycle, 3 gated cycles,
# then probe + burst on the final promoted engine.
source /results/f10/f10lib.sh
export F10_PYCACHE=${F10_PYCACHE:-/results/pycache}
export FOUNDRY_TOML_EXTRA=${FOUNDRY_TOML_EXTRA:-$'skip_jit_warmup = true\nskip_profile_run = true'}
bash $F/setup031.sh || exit 1
cat $F/foundry-load.toml
sed 's/model="m"/model="glm"/' /results/probe.py > $F/probe_glm.py
python3 -u $F/probe_glm.py --label primary --out $F/probe-primary.json --url http://127.0.0.1:8000 > $F/logs/probe-primary.out 2>&1; tail -1 $F/logs/probe-primary.out
[ -z "$NO_BASE" ] && SEED=20261107 bash $F/arm031.sh base-c1 none 0.2
bash $F/chain031g.sh ${NCYC:-3}
P=$(cat /tmp/f10_target); echo "final serving port $P"
python3 -u $F/probe_glm.py --label promoted --out $F/probe-promoted.json --url http://127.0.0.1:$P > $F/logs/probe-promoted.out 2>&1; tail -1 $F/logs/probe-promoted.out
python3 /results/compare.py $F/probe-primary.json $F/probe-promoted.json 2>&1 | tail -20
python3 -u $F/burst.py $P
grep -h "Workspace is locked\|AssertionError" $F/arms/*/*.log $F/arms/*.log 2>/dev/null | head -3
echo "skip lines in released spares: $(grep -h 'skipped warmup_kernels\|skipped profile_run\|lean: skipped' $F/arms/*.log $F/arms/*/*.log 2>/dev/null | sort | uniq -c | head)"
echo RUN_COMBINED_DONE
