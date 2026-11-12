#!/bin/bash
# deploy_startup.sh: GMS (this worktree), load-diff scripts, Foundry overlay and the fig10 harness into pod $P.
NS="-n <namespace>"; P=${P:-rp-ldf-u1}; W=$HOME/dynamo-gms-v031-startup; S=$W/v031-scripts/load-diff; L=$HOME/load-diff-20261111
(cd $W/lib && tar czf /tmp/gms-st.tgz --exclude=__pycache__ gpu_memory_service)
(cd $W/v031-foundry/overlay && tar czf /tmp/overlay-st.tgz --exclude=__pycache__ vllm)
(cd $W/v031-scripts/fig10 && tar czf /tmp/f10-st.tgz --exclude=__pycache__ --exclude=snapshot-sbs *.sh *.py)
kubectl $NS cp /tmp/gms-st.tgz $P:/tmp/gms.tgz; kubectl $NS cp /tmp/overlay-st.tgz $P:/tmp/overlay.tgz; kubectl $NS cp /tmp/f10-st.tgz $P:/tmp/f10.tgz
kubectl $NS cp $HOME/replenish-profile-20261107/common/CORPUS.json $P:/tmp/CORPUS.json
kubectl $NS cp $L/results-83/sizes83.txt $P:/tmp/sizes.txt
kubectl $NS exec $P -- bash -c 'mkdir -p /home/dynamo/gms-v031 /results/scripts /results/f10 && cd /home/dynamo/gms-v031 && tar xzf /tmp/gms.tgz && cp /usr/local/lib/python3.12/dist-packages/gpu_memory_service/client/torch/extensions/_allocator_ext*.so gpu_memory_service/client/torch/extensions/ && cd /results/vllm-site && tar xzf /tmp/overlay.tgz && cd /results/f10 && tar xzf /tmp/f10.tgz && cp /tmp/CORPUS.json /tmp/sizes.txt /results/f10/'
for f in $S/*.sh $S/engine.py $L/compare.py $L/{run,cmd,env,start_gms,kill_engines}.sh; do kubectl $NS cp $f $P:/results/scripts/$(basename $f); done
echo deployed
