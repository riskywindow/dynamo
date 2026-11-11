#!/bin/bash
# kill_engines.sh TAG...  : kill those engines' process groups; "all" kills every vLLM engine (not the GMS server)
for TAG in "$@"; do
  if [ "$TAG" = all ]; then pkill -9 -f '[V]LLM::'; pkill -9 -f '[e]ngine.py'; continue; fi
  PG=$(cat /results/ctl/$TAG.pgid 2>/dev/null) && kill -9 -- -$PG 2>/dev/null && echo "killed $TAG pgid $PG"
done
sleep 5; nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
