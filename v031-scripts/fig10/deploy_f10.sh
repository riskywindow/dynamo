#!/bin/bash
# deploy_f10.sh: push GMS + load-diff scripts + Foundry overlay (from worktree $W) and the fig10 harness into pod $P.
NS="-n <namespace>"; P=${P:?set P}; W=${W:-$(cd "$(dirname "$0")/../.." && pwd)}
sed "s|^NS=.*|NS=\"$NS\"; P=$P; W=$W; S=\$W/v031-scripts/load-diff|" $W/v031-scripts/load-diff/deploy_lean.sh > /tmp/deploy_lean_$P.sh
bash /tmp/deploy_lean_$P.sh
kubectl $NS exec $P -- mkdir -p /results/f10
for f in $W/v031-scripts/fig10/*.sh $W/v031-scripts/fig10/*.py; do kubectl $NS cp $f $P:/results/f10/$(basename $f); done
kubectl $NS cp $HOME/replenish-profile-20261107/common/CORPUS.json $P:/results/f10/CORPUS.json
kubectl $NS cp $HOME/load-diff-20261111/results-sbs/sizes.txt $P:/results/f10/sizes.txt
echo deployed_f10
