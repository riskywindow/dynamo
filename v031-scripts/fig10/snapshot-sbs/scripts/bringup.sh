#!/bin/bash
# fig10-eval copy of snapshot-gms-v1-r0.2/scripts/bringup.sh: source ready -> PodSnapshot -> primary restore -> gate ->
# shadow1 restore. Records times; enforces nothing. (infra/frontend/killer and the source were applied by hand first;
# their apply times are in setup/setup_times.txt.)
NS=<namespace>; P=rp-snap-sbs; D=$HOME/load-diff-20261111/sbs-snap
T=$D/setup/setup_times.txt; k() { kubectl -n $NS "$@" 2> >(grep -v BCID >&2); }
k wait --for=condition=Ready pod/$P-source --timeout=1500s; echo "source_ready_seen $(date +%s.%N)" >> $T
timeout 30 kubectl -n $NS exec $P-infra -c gms -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > $D/setup/nvsmi_before_snapshot.csv 2>&1
echo "snap_apply $(date +%s.%N)" >> $T; k apply -f $D/yaml/30-podsnapshot.yaml
k wait --for=condition=Ready podsnapshot/$P-snap --timeout=900s; echo "snap_ready_seen $(date +%s.%N)" >> $T
k get podsnapshot $P-snap -o yaml > $D/setup/logs/podsnapshot.yaml
k logs --timestamps $P-source -c main > $D/setup/logs/source.log 2>&1
echo "primary_apply $(date +%s.%N)" >> $T; k apply -f $D/yaml/40-primary.yaml
until k logs $P-primary -c main 2>/dev/null | grep -q 'Registering model'; do sleep 1; done; echo "primary_registering_seen $(date +%s.%N)" >> $T
until timeout 60 kubectl -n $NS exec $P-frontend -- curl -s localhost:8000/v1/models 2>/dev/null | grep -q glm; do sleep 2; done
timeout 300 bash $D/scripts/gate.sh > $D/setup/gate_primary.txt 2>&1; echo "gate_done $(date +%s.%N)" >> $T
echo "shadow1_apply $(date +%s.%N)" >> $T; k apply -f $D/yaml/41-shadow1.yaml
until k logs $P-shadow1 -c main 2>/dev/null | grep -q 'waiting for lock'; do sleep 1; done; echo "shadow1_parked_seen $(date +%s.%N)" >> $T
timeout 30 kubectl -n $NS exec $P-infra -c gms -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > $D/setup/nvsmi_primary_plus_shadow1.csv 2>&1
echo BRINGUP_DONE
