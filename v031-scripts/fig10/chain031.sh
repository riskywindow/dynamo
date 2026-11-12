#!/bin/bash
# chain031.sh KIND NCYC [RATE] [DUR]: the Snapshot arm's cycle structure on the GMS path (KIND = foundry | native).
# Needs GMS + weights published (setup031.sh; for foundry also the matched SAVE). If nothing serves on :8000, a KIND
# shadow is started there and promoted first, so every serving engine in the chain is of the same kind (except a live
# primary, which is used as the cycle-1 serving engine when present).
# Start: one parked KIND shadow on :8001. Cycle k (seed 20261107 + 101*(k-1)): load through proxy :9000 (Poisson RATE,
# DUR s of arrivals); at t0+120 s SIGKILL the serving engine, point the proxy at the parked shadow and wake it
# (promotion), then start a replacement KIND shadow on the freed port and park it (replenishment). Next cycle: the
# promoted engine serves and the replacement is the parked shadow. Events -> $F/arms/<KIND>-c<k>/events.txt
source /results/f10/f10lib.sh
KIND=$1; NC=$2; RATE=${3:-0.2}; DUR=${4:-420}; TK=120
TOML=""; [ "$KIND" = foundry ] && TOML=$F/foundry-load.toml
mkdir -p $F/arms
SPORT=8000; SLOG=$F/logs/primary.log; PPORT=8001; PLOG=$F/arms/$KIND-init-shadow.log
if ! ready 8000; then
  SLOG=$F/arms/$KIND-serving0.log; bash $F/serve031.sh shadow 8000 $SLOG $TOML; wait_ready 8000 $SLOG || exit 1
  curl -sf -X POST "localhost:8000/sleep?level=1" > /dev/null; curl -sf -X POST localhost:8000/wake_up > /dev/null
  echo "serving engine: promoted $KIND shadow on :8000"
fi
bash $F/serve031.sh shadow $PPORT $PLOG $TOML; wait_ready $PPORT $PLOG || exit 1
curl -sf -X POST "localhost:$PPORT/sleep?level=1" > /dev/null && echo "initial $KIND shadow parked on :$PPORT"
pkill -f "[p]roxy.py 9000"; echo $SPORT > /tmp/f10_target
setsid python3 $F/proxy.py 9000 /tmp/f10_target > /dev/null 2>&1 < /dev/null &
sleep 1
for k in $(seq $NC); do
  SEED=$((20261107 + 101 * (k - 1))); O=$F/arms/$KIND-c$k; rm -rf $O; mkdir -p $O
  ev() { echo "$1 $(date +%s.%N)" >> $O/events.txt; }
  echo "serving=$SPORT $SLOG parked=$PPORT $PLOG seed=$SEED" > $O/roles.txt
  curl -sf -X POST localhost:$SPORT/reset_prefix_cache > /dev/null
  rm -f /tmp/loadgen.stop
  python3 -u $F/loadgen_perchunk.py --url http://127.0.0.1:9000 --model glm --corpus $F/CORPUS.json --rate $RATE \
    --duration $DUR --seed $SEED --warmup 2 --out $O/requests.jsonl > $O/loadgen.out 2>&1 &
  LG=$!
  for i in $(seq 600); do [ -f $O/loadgen_start.json ] && break; sleep 0.5; done
  T0=$(python3 -c "import json;print(json.load(open('$O/loadgen_start.json'))['t0'])")
  python3 -c "import time; time.sleep(max(0, $T0 + $TK - time.time()))"
  ev kill; kill -9 -- -$(cat $SLOG.pgid) 2>/dev/null; pkill -9 -f "[v]llm serve.*--port $SPORT"; echo $PPORT > /tmp/f10_target
  curl -sf -X POST localhost:$PPORT/wake_up > /dev/null && ev woken || ev wake_failed
  for i in $(seq 240); do curl -sf localhost:$PPORT/v1/models > /dev/null && break; sleep 0.25; done; ev promoted
  RLOG=$O/shadow.log; ev rep_start; bash $F/serve031.sh shadow $SPORT $RLOG $TOML
  if wait_ready $SPORT $RLOG; then ev rep_ready
    curl -sf -X POST "localhost:$SPORT/sleep?level=1" > /dev/null && ev rep_parked || ev park_failed
  else ev rep_failed; fi
  wait $LG; ev load_done
  echo "$KIND c$k: $(grep -c . $O/requests.jsonl) requests, errors $(grep -c '"error"' $O/requests.jsonl)"; cat $O/events.txt
  grep -q rep_parked $O/events.txt || { echo "no parked replacement; stopping the chain"; break; }
  T=$PPORT; L=$PLOG; PPORT=$SPORT; PLOG=$RLOG; SPORT=$T; SLOG=$L
done
