#!/bin/bash
# arm031.sh NAME SHADOW RATE [DUR] [T_EVENT]: one measured arm against the live primary (:8000).
#   SHADOW=none: no-failure baseline. foundry: lean Foundry LOAD shadow. native: native-capture shadow.
#   Load: per-chunk loadgen, 32K/1K, Poisson RATE, fixed seed. At t0+T_EVENT the shadow is launched on :8001,
#   parked with POST /sleep?level=1 once it answers, and killed when the load ends. Events -> $O/events.txt
source /results/f10/f10lib.sh
N=$1; SH=$2; RATE=$3; DUR=${4:-420}; TE=${5:-120}; O=$F/arms/$N; rm -rf $O; mkdir -p $O
curl -sf -X POST localhost:8000/reset_prefix_cache > /dev/null && echo "prefix cache reset"
rm -f /tmp/loadgen.stop
python3 -u $F/loadgen_perchunk.py --url http://127.0.0.1:8000 --model glm --corpus $F/CORPUS.json --rate $RATE \
  --duration $DUR --seed ${SEED:-20261107} --warmup 2 --out $O/requests.jsonl > $O/loadgen.out 2>&1 &
LG=$!
for i in $(seq 600); do [ -f $O/loadgen_start.json ] && break; sleep 0.5; done
T0=$(python3 -c "import json;print(json.load(open('$O/loadgen_start.json'))['t0'])")
python3 -c "import time; time.sleep(max(0, $T0 + $TE - time.time()))"
ev() { echo "$1 $(date +%s.%N)" >> $O/events.txt; }
if [ "$SH" != none ]; then
  TOML=""; [ "$SH" = foundry ] && TOML=$F/foundry-load.toml
  ev launch; bash $F/serve031.sh shadow 8001 $O/shadow.log $TOML
  if wait_ready 8001 $O/shadow.log; then ev ready
    curl -sf -X POST "localhost:8001/sleep?level=1" > /dev/null && ev parked || ev park_failed
  else ev shadow_failed; fi
fi
wait $LG; ev load_done
[ "$SH" != none ] && { kill_engine $O/shadow.log; pkill -9 -f "[v]llm serve.*--port 8001"; sleep 5; }
echo "arm $N done: $(grep -c . $O/requests.jsonl) requests, errors $(grep -c '"error"' $O/requests.jsonl)"; cat $O/events.txt
