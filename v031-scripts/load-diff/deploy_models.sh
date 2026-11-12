#!/bin/bash
# deploy_lean.sh: push GMS (worktree), load-diff scripts and the changed Foundry overlay files into pod $P
NS="-n <namespace>"; P=${P:-rp-ldf-c}; W=$HOME/dynamo-gms-v031-models; S=$W/v031-scripts/load-diff
(cd $W/lib && tar czf /tmp/gms-lean.tgz --exclude=__pycache__ gpu_memory_service)
(cd $W/v031-foundry/overlay && tar czf /tmp/overlay-lean.tgz --exclude=__pycache__ vllm)
kubectl $NS cp /tmp/gms-lean.tgz $P:/tmp/gms.tgz
kubectl $NS cp /tmp/overlay-lean.tgz $P:/tmp/overlay.tgz
kubectl $NS exec $P -- bash -c 'mkdir -p /home/dynamo/gms-v031 /results/scripts /results/ldf && cd /home/dynamo/gms-v031 && tar xzf /tmp/gms.tgz && cp /usr/local/lib/python3.12/dist-packages/gpu_memory_service/client/torch/extensions/_allocator_ext*.so gpu_memory_service/client/torch/extensions/ && cd /results/vllm-site && tar xzf /tmp/overlay.tgz'
for f in $S/*.sh $S/engine.py $HOME/load-diff-20261111/compare.py $HOME/load-diff-20261111/{run,cmd,env,start_gms,kill_engines}.sh; do kubectl $NS cp $f $P:/results/scripts/$(basename $f); done
for f in $HOME/load-diff-20261111/prompts*.json; do kubectl $NS cp $f $P:/results/$(basename $f); done
echo deployed
