#!/bin/bash
# Snapshot + GMS v1 arm on the per-chunk harness (2026-11-11): the same workload as the lean/native PNNP chains.
#   per-chunk loadgen (validation/pod/loadgen_perchunk.py, staged by stage_frontend.sh), 0.2 req/s, 420 s of arrivals,
#   SIGKILL at loadgen t0+120 s, per-cycle seeds = the lean chains' (20261107 + 101*k): c1 20261208 ... c4 20261511.
# Rotation as in fig10-eval: the replacement restored in cycle k is the parked shadow of cycle k+1. No baseline (the
# lean/native chains use each cycle's own pre-failure window). Stops if a cycle leaves no parked shadow.
#   c1: kill primary, shadow1 promotes, restore shadow2      c2: kill shadow1, shadow2 promotes, restore shadow3
#   c3: kill shadow2, shadow3 promotes, restore shadow4      c4: kill shadow3, shadow4 promotes, restore shadow5
# Usage: setsid nohup bash orchestrate.sh [c1 c2 c3 c4] > orchestrate.stdout 2>&1 < /dev/null &
set -u
NS=<namespace>; P=rp-snap-sbs
D=$HOME/load-diff-20261111/sbs-snap
C=$HOME/replenish-profile-20261107/common
O=$D/orchestrate.log
log() { echo "$1 $(date +%s.%N) $(date -u +%T) ${2:-}" | tee -a $O; }
python3 -c "import json;json.dump({'rate':0.2,'source':'fixed: same rate as the lean/native chains'},open('$D/rate_used.json','w'))"

analyze() {  # <name>
  local N=$1 M=""
  [ -s $D/$N/metrics.jsonl ] && grep -q '"ok": true' $D/$N/metrics.jsonl && M="--metrics $D/$N/metrics.jsonl"
  python3 $C/analyze.py --requests $D/$N/requests.jsonl --events $D/$N/events.json --out $D/$N/summary.json \
    --label snapshot-$N $M > $D/$N/analyze.out 2>&1
  log analyzed "$N rc=$? metrics=${M:+yes}"
}

declare -A SPEC=(   # side-by-side 2026-11-12: seeds 20261107 + 101*(k-1), same as the GMS arms
  [c1]="primary shadow1 shadow2 42-shadow2.yaml 20261107"
  [c2]="shadow1 shadow2 shadow3 43-shadow3.yaml 20261208"
  [c3]="shadow2 shadow3 shadow4 44-shadow4.yaml 20261309"
)
CYCLES=${*:-c1 c2 c3}
for N in $CYCLES; do
  set -- ${SPEC[$N]}; SERV=$1; PARK=$2; REP=$3; Y=$4; SEED=$5
  log ${N}_start "kill $SERV, promote $PARK, restore $REP, seed $SEED"
  bash $D/scripts/run.sh $N kill $SERV $PARK $Y $SEED > $D/$N.stdout 2>&1
  bash $D/scripts/collect.sh $N $SERV $PARK $REP > /dev/null 2>&1
  python3 $D/scripts/make_events.py $D/$N > $D/$N/make_events.out 2>&1
  python3 - "$D/$N/events.json" "$N" "$SERV" "$PARK" "$REP" "$SEED" <<'EOF'
import json, sys
p, cyc, victim, survivor, rep, seed = sys.argv[1:]
e = json.load(open(p))
e.update(cycle=cyc, arm='snapshot', victim=victim, survivor=survivor, replacement=rep, seed=int(seed))
json.dump(e, open(p, 'w'), indent=1)
EOF
  analyze $N
  grep -q '^kill_exec_returned .*rc=0' $D/$N/failover_events.txt && kubectl -n $NS delete pod $P-$SERV --wait=false >> $O 2>&1
  if ! grep -q '^rep_parked_seen' $D/$N/failover_events.txt; then
    log STOP "$N: no parked replacement; next cycle would have no shadow. Stopping for diagnosis."
    exit 1
  fi
  sleep 20
done
log ALL_DONE
