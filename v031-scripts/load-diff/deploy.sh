#!/bin/bash
# deploy.sh: push the patched GMS package + scripts into the pod
NS="-n <namespace>"; P=${P:-rp-ldiff-a}; D=$HOME/load-diff-20261111
(cd $HOME/dynamo-gms-v031/lib && tar czf /tmp/gms-ldiff.tgz --exclude=__pycache__ gpu_memory_service)
kubectl $NS cp /tmp/gms-ldiff.tgz $P:/tmp/gms.tgz
kubectl $NS exec $P -- bash -c 'cd /home/dynamo/gms-v031 && tar xzf /tmp/gms.tgz && cp /usr/local/lib/python3.12/dist-packages/gpu_memory_service/client/torch/extensions/_allocator_ext*.so gpu_memory_service/client/torch/extensions/'
for f in $D/*.sh $D/engine.py $D/compare.py; do kubectl $NS cp $f $P:/results/scripts/$(basename $f); done
echo deployed
