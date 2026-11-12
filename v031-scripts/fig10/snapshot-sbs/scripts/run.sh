#!/bin/bash
# One measured run of the Snapshot arm (fig10-eval). Records everything, enforces nothing.
#   run.sh <name> baseline <serving> <parked>                         no kill, no replenishment
#   run.sh <name> kill     <serving> <parked> <replacement-yaml>      SIGKILL <serving> at t0+120 s, <parked> promotes,
#                                                                     restore <replacement-yaml> as soon as it is awake
# <serving>/<parked> are pod suffixes (primary, shadow1, ...). Rate comes from $D/rate_used.json.
# Adapted from snapshot-gms-v1-r0.2/scripts/failover.sh (same killer-pod SIGKILL, same promotion/park log markers).
set -u
NAME=$1; MODE=$2; SERV=$3; PARK=$4; REPY=${5:-}; SEED=${6:?seed (6th arg; the per-chunk loadgen would default to 20261107 in /tmp/work)}
NS=<namespace>; P=rp-snap-sbs
D=$HOME/load-diff-20261111/sbs-snap; OUT=$D/$NAME; EV=$OUT/failover_events.txt
mkdir -p $OUT/logs
DUR=${DUR:-420}  # same 420 s cycle as the lean/native chains (FIG10_CYCLE_S=420)
RATE=$(python3 -c "import json;print(json.load(open('$D/rate_used.json'))['rate'])")
k() { kubectl -n $NS "$@" 2> >(grep -v BCID >&2); }
log() { echo "$1 $(date +%s.%N) ${2:-}" | tee -a $EV; }
fe() { timeout 60 kubectl -n $NS exec $P-frontend -c frontend -- bash -c "$1" 2> >(grep -v BCID >&2); }

log script_start "name=$NAME seed=$SEED mode=$MODE serving=$SERV parked=$PARK replacement=$REPY rate=$RATE"
SIP=$(k get pod $P-$SERV -o jsonpath='{.status.podIP}'); PIP=$(k get pod $P-$PARK -o jsonpath='{.status.podIP}')
log pod_ips "serving=$SIP parked=$PIP"
# Clock check (record only): b - node = return latency of kubectl exec; true node-local offset is within +-that.
python3 - >> $EV <<EOF
import subprocess,time
best=None
for i in range(3):
    a=time.time(); n=float(subprocess.check_output(["kubectl","-n","$NS","exec","$P-killer","--","date","+%s.%N"],stderr=subprocess.DEVNULL)); b=time.time()
    if best is None or b-a<best[0]: best=(b-a,n-a,b-n)
print("clock_check %.3f rtt=%.3f node_minus_send=%.3f recv_minus_node=%.3f" % (time.time(),*best))
EOF

# dmon (infra pod, sees the 4 GPUs), metrics scrape and loadgen (frontend pod), all detached from the exec session.
timeout 30 kubectl -n $NS exec $P-infra -c gms -- bash -c "mkdir -p /results/$NAME; nohup timeout 1200 nvidia-smi dmon -s u -d 1 -o DT > /results/$NAME/dmon.txt 2>&1 < /dev/null &" 2> >(grep -v BCID >&2)
fe "mkdir -p /results/$NAME; cd /tmp/work; nohup python3 metrics_scrape.py --url http://$SIP:8081/metrics --url http://$PIP:8081/metrics \
  --out /results/$NAME/metrics.jsonl --duration 1200 --stop-file /results/$NAME/STOP_METRICS > /results/$NAME/metrics_scrape.log 2>&1 < /dev/null &"
log monitors_started
fe "cd /tmp/work; nohup python3 loadgen.py --url http://127.0.0.1:8000 --model glm-5.2-nvfp4 --corpus CORPUS.json --rate $RATE \
  --duration $DUR --seed $SEED --warmup 2 --out /results/$NAME/requests.jsonl --stop-file /results/$NAME/STOP_LOADGEN \
  > /results/$NAME/loadgen.log 2>&1 < /dev/null &"
log loadgen_launched
t0=""
for i in $(seq 1 600); do
  t0=$(fe "python3 -c 'import json;print(json.load(open(\"/results/$NAME/loadgen_start.json\"))[\"t0\"])'" 2>/dev/null) && break
  t0=""; sleep 1
done
log t0 "$t0"
[ -z "$t0" ] && { log t0_missing "loadgen_start.json never appeared; no kill"; MODE=nokill; }
[ "$MODE" = kill ] && log parked_check "$(timeout 20 kubectl -n $NS logs $P-$PARK -c main 2>/dev/null | grep -c 'waiting for lock') 'waiting for lock' lines in $PARK"

if [ "$MODE" = kill ]; then
  target_ns=$(python3 -c "print(int(($t0+120)*1e9))")
  # SIGKILL from the hostPID, AppArmor-unconfined killer pod (exec'd shells in engine pods are AppArmor-confined and cannot
  # signal the agent-restored, unconfined engine processes). PIDs resolved before the deadline via the pod-UID cgroup.
  PUID=$(k get pod $P-$SERV -o jsonpath='{.metadata.uid}' | tr '-' '_')
  timeout 300 kubectl -n $NS exec $P-killer -- bash -c "PIDS=\$(for f in \$(grep -l 'pod$PUID' /proc/[0-9]*/cgroup 2>/dev/null); do q=\$(echo \$f | cut -d/ -f3); case \$(cat /proc/\$q/comm 2>/dev/null) in python3|VLLM*) echo \$q;; esac; done); \
    echo pids \$PIDS; ps -o pid,ppid,stat,comm -p \$(echo \$PIDS | tr ' ' ','); while [ \$(date +%s%N) -lt $target_ns ]; do sleep 0.002; done; T1=\$(date +%s.%N); kill -9 \$PIDS; RC=\$?; T2=\$(date +%s.%N); \
    echo signal_sent \$T1 rc=\$RC; echo kill_returned \$T2; \
    for i in \$(seq 1 20000); do ps -o stat= -p \$(echo \$PIDS | tr ' ' ',') 2>/dev/null | grep -qv Z || break; sleep 0.001; done; echo all_exited_or_zombie \$(date +%s.%N); ps -o pid,stat,comm -p \$(echo \$PIDS | tr ' ' ',')" > $OUT/kill_output.txt 2>&1
  log kill_exec_returned "$(grep signal_sent $OUT/kill_output.txt)"
  grep -q 'signal_sent .* rc=0' $OUT/kill_output.txt || { log kill_failed "see kill_output.txt"; MODE=killfailed; }
fi
if [ "$MODE" = kill ]; then
  timeout 600 bash -c "kubectl -n $NS logs -f --timestamps $P-$PARK -c main 2>/dev/null | grep -m1 -E '\[Shadow\] Engine awake'" > $OUT/promo_line.txt
  log promoted_seen "$(cut -c1-120 $OUT/promo_line.txt)"
  [ -s $OUT/promo_line.txt ] || log promotion_not_seen "no '[Shadow] Engine awake' in $PARK within 600 s; restoring the replacement anyway"
  log rep_start
  k apply -f $D/yaml/$REPY >> $EV 2>&1
  log rep_applied
  RPOD=$(grep -m1 '^  name:' $D/yaml/$REPY | awk '{print $2}')
  for i in $(seq 1 1200); do
    l=$(timeout 20 kubectl -n $NS logs --timestamps $RPOD -c main 2>/dev/null | grep -m1 -E 'waiting for lock')
    [ -n "$l" ] && { log rep_parked_seen "$(echo "$l" | cut -c1-40)"; break; }
    sleep 0.5
  done
  grep -q '^rep_parked_seen' $EV || log rep_not_parked "replacement $RPOD not parked within the poll window"
  k get pod $RPOD -o yaml > $OUT/logs/pod-replacement.live.yaml
fi

# Let the client finish (720 s of arrivals, then in-flight requests drain).
until fe "grep -q '^done' /results/$NAME/loadgen.log" 2>/dev/null; do sleep 15; done
log loadgen_done
sleep 5
fe "touch /results/$NAME/STOP_METRICS"
timeout 30 kubectl -n $NS exec $P-infra -c gms -- bash -c "pkill -f 'nvidia-smi dmon' ; true" 2> >(grep -v BCID >&2)
log monitors_stopped

# Copy out (frontend: client + metrics; infra: dmon).
for f in requests.jsonl warmup.jsonl loadgen_start.json loadgen.log metrics.jsonl metrics_scrape.log; do
  fe "cat /results/$NAME/$f" > $OUT/$f
done
timeout 60 kubectl -n $NS exec $P-infra -c gms -- cat /results/$NAME/dmon.txt > $OUT/dmon.txt 2> >(grep -v BCID >&2)
log copied
echo RUN_DONE
