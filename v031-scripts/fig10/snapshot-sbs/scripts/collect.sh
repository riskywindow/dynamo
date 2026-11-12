#!/bin/bash
# Copy logs for one run into <name>/logs (best effort, never fails the run).
#   collect.sh <name> <serving> <parked> [<replacement>]
set -u
NAME=$1; SERV=$2; PARK=$3; REP=${4:-}
NS=<namespace>; P=rp-snap-sbs
D=$HOME/load-diff-20261111/sbs-snap; L=$D/$NAME/logs; mkdir -p $L
SINCE=$(python3 -c "import datetime as d;print(d.datetime.fromtimestamp(float(open('$D/$NAME/failover_events.txt').readline().split()[1])-120,d.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'))")
q() { "$@" 2> >(grep -v BCID >&2); }
q kubectl -n $NS logs --timestamps $P-$SERV -c main > $L/serving-$SERV.log
q kubectl -n $NS logs --timestamps $P-$PARK -c main > $L/parked-$PARK.log
[ -n "$REP" ] && q kubectl -n $NS logs --timestamps $P-$REP -c main > $L/replacement-$REP.log
for r in $SERV $PARK $REP; do q kubectl -n $NS get pod $P-$r -o yaml > $L/pod-$r.live.yaml; done
q kubectl -n $NS logs --timestamps $P-frontend -c frontend --since-time=$SINCE > $L/frontend.log
q kubectl -n $NS logs --timestamps $P-infra -c gms > $L/infra-gms.log
q kubectl -n $NS logs --timestamps $P-infra -c etcd --since-time=$SINCE > $L/infra-etcd.log
AG=$(kubectl -n snapshot get pods -o wide --no-headers | awk '/snapshot-agent/ && /xjjk/ {print $1}')
q kubectl -n snapshot logs --timestamps $AG -c agent --since-time=$SINCE > $L/agent.log
q kubectl -n snapshot logs --timestamps $AG -c pagebroker --since-time=$SINCE > $L/pagebroker.log
q kubectl -n $NS get events --sort-by=.lastTimestamp > $L/events-ns.txt
timeout 30 kubectl -n $NS exec $P-infra -c gms -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > $L/nvsmi_after.csv 2>/dev/null
echo "collected $(date +%s.%N) since=$SINCE agent=$AG" >> $D/$NAME/failover_events.txt
