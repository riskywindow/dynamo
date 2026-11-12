#!/bin/bash
# chain031g.sh NCYC [RATE] [DUR] (K0=<first cycle index>, default 1; cycle k uses seed 20261107 + 101*(k-1)): chain031.sh foundry cycles with a pre-started, gated spare (F10_GATE).
# Ports rotate between serving, parked (sleeping lean-Foundry shadow) and gated (lean-Foundry shadow held after weight
# attach). Cycle k (seed 20261107 + 101*(k-1)): at t0+120 s SIGKILL serving; proxy -> parked, wake it (promotion);
# release the gated spare's gate (rep_start), wait until it serves, park it (rep_parked); then pre-start a new gated
# spare on the freed port (pre_start .. pre_held = all ranks holding). Events -> $F/arms/gated-c<k>/events.txt.
# Needs setup031.sh (GMS, primary on :8000 as the cycle-1 serving engine, matched lean SAVE).
source /results/f10/f10lib.sh
NC=$1; RATE=${2:-0.2}; DUR=${3:-420}; TK=120; TOML=$F/foundry-load.toml
mkdir -p $F/arms
held() { [ "$(grep -c 'startup gate: holding' $1 2>/dev/null)" -ge 4 ]; }
start_gated() {  # PORT LOG GATEFILE
  rm -f $3; F10_GATE=$3 bash $F/serve031.sh shadow $1 $2 $TOML
  for i in $(seq 1800); do held $2 && return 0; grep -qE "Traceback|Segfault" $2 && return 1; sleep 0.25; done; return 1; }
SPORT=8000; SLOG=$F/logs/primary.log
PPORT=8001; PLOG=$F/arms/gated-init-parked.log
GPORT=8002; GLOG=$F/arms/gated-init-gated.log; GGATE=/tmp/gate-init
ready 8000 || { echo "no primary on :8000"; exit 1; }
bash $F/serve031.sh shadow $PPORT $PLOG $TOML; wait_ready $PPORT $PLOG || exit 1
curl -sf -X POST "localhost:$PPORT/sleep?level=1" > /dev/null && echo "initial shadow parked on :$PPORT"
start_gated $GPORT $GLOG $GGATE || { echo "initial gated spare failed"; exit 1; }; echo "initial spare gated on :$GPORT"
pkill -f "[p]roxy.py 9000"; echo $SPORT > /tmp/f10_target
setsid python3 $F/proxy.py 9000 /tmp/f10_target > /dev/null 2>&1 < /dev/null &
sleep 1
K0=${K0:-1}
for k in $(seq $K0 $((K0 + NC - 1))); do
  SEED=$((20261107 + 101 * (k - 1))); O=$F/arms/gated-c$k; rm -rf $O; mkdir -p $O
  ev() { echo "$1 $(date +%s.%N)" >> $O/events.txt; }
  echo "serving=$SPORT parked=$PPORT gated=$GPORT seed=$SEED" > $O/roles.txt
  curl -sf -X POST localhost:$SPORT/reset_prefix_cache > /dev/null
  python3 -u $F/loadgen_perchunk.py --url http://127.0.0.1:9000 --model glm --corpus $F/CORPUS.json --rate $RATE \
    --duration $DUR --seed $SEED --warmup 2 --out $O/requests.jsonl > $O/loadgen.out 2>&1 &
  LG=$!
  for i in $(seq 600); do [ -f $O/loadgen_start.json ] && break; sleep 0.5; done
  T0=$(python3 -c "import json;print(json.load(open('$O/loadgen_start.json'))['t0'])")
  python3 -c "import time; time.sleep(max(0, $T0 + $TK - time.time()))"
  ev kill; kill -9 -- -$(cat $SLOG.pgid) 2>/dev/null; pkill -9 -f "[v]llm serve.*--port $SPORT"; echo $PPORT > /tmp/f10_target
  curl -sf -X POST localhost:$PPORT/wake_up > /dev/null && ev woken || ev wake_failed
  for i in $(seq 240); do curl -sf localhost:$PPORT/v1/models > /dev/null && break; sleep 0.25; done; ev promoted
  ev rep_start; touch $GGATE
  if wait_ready $GPORT $GLOG; then ev rep_ready
    curl -sf -X POST "localhost:$GPORT/sleep?level=1" > /dev/null && ev rep_parked || ev park_failed
  else ev rep_failed; fi
  NLOG=$O/gated-next.log; NGATE=/tmp/gate-c$k
  ev pre_start; start_gated $SPORT $NLOG $NGATE && ev pre_held || ev pre_failed
  wait $LG; ev load_done
  echo "gated c$k: $(grep -c . $O/requests.jsonl) requests, errors $(grep -c '"error"' $O/requests.jsonl)"; cat $O/events.txt
  grep -q rep_parked $O/events.txt && grep -q pre_held $O/events.txt || { echo "chain broken; stopping"; break; }
  # rotate: promoted -> serving, released spare -> parked, new gated spare -> gated
  NS_PORT=$PPORT; NS_LOG=$PLOG; PPORT=$GPORT; PLOG=$GLOG; GPORT=$SPORT; GLOG=$NLOG; GGATE=$NGATE; SPORT=$NS_PORT; SLOG=$NS_LOG
done
