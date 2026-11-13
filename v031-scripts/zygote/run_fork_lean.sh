#!/bin/bash
# run_fork_lean.sh MODE [TAG]: lean Foundry SAVE/LOAD with matched process start on both sides, no load, 83 sizes.
#   MODE=spawn: reference (spawned engine core + workers). MODE=fork: F10_FORK=1 + cudaclean (forked engine core and
#   workers; Foundry's hook preloaded for the whole tree). MODE=zygote: SAVE forked as in fork, LOAD shadow forked from
#   the zygote (F10_ZYGOTE, see zygote.py / zystart.sh).
# Steps: bytecode cache (once), setup031.sh (GMS, primary, matched SAVE, all four lean switches), one cold lean LOAD
# shadow next to the live primary (startup031.sh cold) -> phase timeline. Logs stay in /results/f10/logs.
source /results/f10/f10lib.sh
MODE=$1; TAG=${2:-$MODE}
export F10_PYCACHE=${F10_PYCACHE:-/results/pycache}
export FOUNDRY_TOML_EXTRA=${FOUNDRY_TOML_EXTRA:-$'skip_jit_warmup = true\nskip_profile_run = true'}
if [ ! -d $F10_PYCACHE ]; then t=$(date +%s)
  PYTHONPYCACHEPREFIX=$F10_PYCACHE python3 -m compileall -q -j 32 /usr/local/lib/python3.12/dist-packages /results/vllm-site \
    $HOME/.local/lib/python3.12/site-packages /home/dynamo/gms-v031 /results/zy > /dev/null 2>&1
  echo "pycache filled in $(( $(date +%s) - t )) s, $(du -sh $F10_PYCACHE | cut -f1)"; fi
if [ "$MODE" != spawn ]; then
  export F10_FORK=1 RP_CUDA_CLEAN_IMPORTS=1 PYTHONPATH=/results/zy/audit${PYTHONPATH:+:$PYTHONPATH}
fi
[ -z "$NO_SETUP" ] && { bash $F/setup031.sh || exit 1; }
grep -h "Graph capturing finished" $F/logs/save.log | sed 's/^/save: /' | head -3
if [ "$MODE" = zygote ]; then
  bash /results/zy/zystart.sh /tmp/zygote.sock /results/f10/logs/zygote-$TAG.log || exit 1
  export F10_ZYGOTE=/tmp/zygote.sock
fi
bash $F/startup031.sh $TAG cold
L=$F/logs/st-$TAG.log
grep -h "Graph capturing finished\|init engine\|Model loading took\|lean: \|skipped warmup_kernels\|skipped profile_run\|RP_CUDACLEAN release\|must use the .spawn" $L \
  | grep -E "TP0|EngineCore|APIServer|RP_CUDACLEAN|spawn" | cut -c1-220 | head -14
echo "restored=$(grep -c 'restored graph descriptor' $L) breakable_loaded=$(grep -c 'breakable LOAD' $L) layout_mismatch=$(grep -ci 'layout mismatch\|mismatch' $L)"
echo RUN_FORK_LEAN_DONE $TAG
