#!/bin/bash
# promote031.sh NAME SHADOW RATE [DUR] [T_EVENT] [T_KILL]: like arm031.sh, but load goes through proxy :9000 and at
# t0+T_KILL the primary is SIGKILLed, the proxy is switched to :8001 and the parked shadow is woken (promotion).
source /results/f10/f10lib.sh
N=$1; SH=$2; RATE=$3; DUR=${4:-480}; TE=${5:-120}; TK=${6:-300}; O=$F/arms/$N; rm -rf $O; mkdir -p $O
echo 8000 > /tmp/f10_target; pkill -f "[p]roxy.py 9000"; setsid python3 $F/proxy.py 9000 /tmp/f10_target > /dev/null 2>&1 < /dev/null &
sleep 1; curl -sf -X POST localhost:8000/reset_prefix_cache > /dev/null && echo "prefix cache reset"
rm -f /tmp/loadgen.stop
python3 -u $F/loadgen_perchunk.py --url http://127.0.0.1:9000 --model glm --corpus $F/CORPUS.json --rate $RATE \
  --duration $DUR --seed 20261107 --warmup 2 --out $O/requests.jsonl > $O/loadgen.out 2>&1 &
LG=$!
for i in $(seq 600); do [ -f $O/loadgen_start.json ] && break; sleep 0.5; done
T0=$(python3 -c "import json;print(json.load(open('$O/loadgen_start.json'))['t0'])")
ev() { echo "$1 $(date +%s.%N)" >> $O/events.txt; }
python3 -c "import time; time.sleep(max(0, $T0 + $TE - time.time()))"
TOML=""; [ "$SH" = foundry ] && TOML=$F/foundry-load.toml
ev launch; bash $F/serve031.sh shadow 8001 $O/shadow.log $TOML
if wait_ready 8001 $O/shadow.log; then ev ready; curl -sf -X POST "localhost:8001/sleep?level=1" > /dev/null && ev parked; else ev shadow_failed; fi
python3 -c "import time; time.sleep(max(0, $T0 + $TK - time.time()))"
ev kill; kill -9 -- -$(cat $F/logs/primary.log.pgid); pkill -9 -f "[v]llm serve.*--port 8000"; echo 8001 > /tmp/f10_target
curl -sf -X POST localhost:8001/wake_up > /dev/null && ev woken || ev wake_failed
for i in $(seq 120); do curl -sf localhost:8001/v1/models > /dev/null && break; sleep 0.5; done; ev serving
wait $LG; ev load_done
echo "arm $N done: $(grep -c . $O/requests.jsonl) requests, errors $(grep -c '"error"' $O/requests.jsonl)"; cat $O/events.txt
