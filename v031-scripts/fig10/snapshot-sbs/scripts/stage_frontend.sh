#!/bin/bash
# Stage the per-chunk loadgen (as loadgen.py), the corpus and the metrics scraper into the frontend pod's /tmp/work.
# Same files as the lean/native chains (run_leanload.sh copies $F/common/* to /test/rp; launch stages the per-chunk
# loadgen as common/loadgen.py). Prints shas so they can be compared with the lean pod's.
NS=<namespace>; P=rp-snap-sbs
LG=${LOADGEN_DIR:-$HOME/lean-load}/validation/pod/loadgen_perchunk.py
C=$HOME/replenish-profile-20261107/common
kubectl -n $NS exec $P-frontend -c frontend -- mkdir -p /tmp/work /results
kubectl -n $NS cp $LG $P-frontend:/tmp/work/loadgen.py -c frontend
kubectl -n $NS cp $C/CORPUS.json $P-frontend:/tmp/work/CORPUS.json -c frontend
kubectl -n $NS cp $C/metrics_scrape.py $P-frontend:/tmp/work/metrics_scrape.py -c frontend
kubectl -n $NS exec $P-frontend -c frontend -- bash -c 'cd /tmp/work && sha256sum loadgen.py CORPUS.json metrics_scrape.py && grep -c _default_seed loadgen.py'
sha256sum $LG $C/CORPUS.json $C/metrics_scrape.py
