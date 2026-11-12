#!/bin/bash
# run_n5.sh: more cycles of the combined arm (gated spare + four matched lean switches + bytecode cache), 83 sizes.
# Fills the bytecode cache, runs setup (GMS, primary, matched SAVE), probes the primary, runs fresh no-failure
# baselines for BASE_K (cycle indices; seed 20261107 + 101*(k-1)), then NCYC gated cycles starting at K0, then
# probes the final promoted engine and compares it with the primary.
source /results/f10/f10lib.sh
export F10_PYCACHE=${F10_PYCACHE:-/results/pycache}
export FOUNDRY_TOML_EXTRA=${FOUNDRY_TOML_EXTRA:-$'skip_jit_warmup = true\nskip_profile_run = true'}
if [ ! -d $F10_PYCACHE ]; then t=$(date +%s)
  PYTHONPYCACHEPREFIX=$F10_PYCACHE python3 -m compileall -q -j 32 /usr/local/lib/python3.12/dist-packages /results/vllm-site \
    $HOME/.local/lib/python3.12/site-packages /home/dynamo/gms-v031 > /dev/null 2>&1
  echo "pycache filled in $(( $(date +%s) - t )) s, $(du -sh $F10_PYCACHE | cut -f1)"; fi
bash $F/setup031.sh || exit 1
sed 's/model="m"/model="glm"/' /results/probe.py > $F/probe_glm.py
python3 -u $F/probe_glm.py --label primary --out $F/probe-primary.json --url http://127.0.0.1:8000 > $F/logs/probe-primary.out 2>&1; tail -1 $F/logs/probe-primary.out
for k in ${BASE_K:-4 5}; do SEED=$((20261107 + 101 * (k - 1))) bash $F/arm031.sh base-c$k none 0.2; done
K0=${K0:-3} bash $F/chain031g.sh ${NCYC:-3}
P=$(cat /tmp/f10_target); echo "final serving port $P"
python3 -u $F/probe_glm.py --label promoted --out $F/probe-promoted.json --url http://127.0.0.1:$P > $F/logs/probe-promoted.out 2>&1; tail -1 $F/logs/probe-promoted.out
python3 /results/compare.py $F/probe-primary.json $F/probe-promoted.json 2>&1 | tail -20
echo RUN_N5_DONE
