#!/bin/bash
# startup031.sh TAG [cold|gate]: one lean-Foundry GMS shadow on :8001 next to the live primary, no load.
# Prints the phase timeline (startup_phases.py). gate: the shadow is pre-started with F10_GATE, held after weight
# attach; its memory is recorded, then the gate is released and release->ready->parked is timed.
# Honors F10_PYCACHE (bytecode cache prefix). Needs setup031.sh (GMS, primary, matched SAVE) first.
source /results/f10/f10lib.sh; TAG=$1; MODE=${2:-cold}
L=$F/logs/st-$TAG.log; G=/tmp/gate-$TAG; rm -f $G $L.events
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ' > $L.mem; echo " before" >> $L.mem
[ "$MODE" = gate ] && export F10_GATE=$G
bash $F/serve031.sh shadow 8001 $L $F/foundry-load.toml
if [ "$MODE" = gate ]; then
  for i in $(seq 1800); do [ "$(grep -c 'startup gate: holding' $L)" -ge 4 ] && break
    grep -qE "Traceback|Segfault" $L && { echo "DIED $L"; exit 1; }; sleep 0.5; done
  sleep 5
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ' >> $L.mem; echo " gated" >> $L.mem
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $L.mem
  echo "release $(date +%s.%N)" >> $L.events; touch $G
fi
wait_ready 8001 $L || exit 1; echo "ready $(date +%s.%N)" >> $L.events
curl -sf -X POST "localhost:8001/sleep?level=1" > /dev/null && echo "parked $(date +%s.%N)" >> $L.events
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ' >> $L.mem; echo " parked" >> $L.mem
python3 $F/startup_phases.py $L | grep -v JSON
python3 - $L <<'PY'
import sys
L=sys.argv[1]; ev={l.split()[0]: float(l.split()[1]) for l in open(L+".events")}; t0=float(open(L+".launch").read())
print("launch->ready %.1f s  launch->parked %.1f s" % (ev["ready"]-t0, ev["parked"]-t0)
      + ("  release->ready %.1f s  release->parked %.1f s" % (ev["ready"]-ev["release"], ev["parked"]-ev["release"]) if "release" in ev else ""))
PY
cat $L.mem | head -3
kill_engine $L; pkill -9 -f "[v]llm serve.*--port 8001"; sleep 5
