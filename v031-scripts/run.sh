#!/bin/bash
# run.sh TAG [env assignments...] -- engine.py args
# Starts one engine in its own session (pgid saved in /results/ctl/TAG.pgid); log at /results/logs/TAG.log
source /results/scripts/env.sh
TAG=$1; shift
ENVS=(); while [ "$1" != "--" ] && [ -n "$1" ]; do ENVS+=("$1"); shift; done; shift
echo "$(date +%s.%N) launch $TAG ${ENVS[*]} -- $*" >> /results/logs/launches.txt
setsid nohup env "${ENVS[@]}" python3 /results/scripts/engine.py --tag $TAG "$@" > /results/logs/$TAG.log 2>&1 < /dev/null &
echo $! > /results/ctl/$TAG.pgid
echo "started $TAG pid/pgid $!"
