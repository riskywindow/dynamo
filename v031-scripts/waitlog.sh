#!/bin/bash
# waitlog.sh TAG PATTERN [timeout_s] : wait until /results/logs/TAG.log matches PATTERN or an error appears
TAG=$1; PAT=$2; TO=${3:-1800}; end=$(( $(date +%s) + TO ))
while [ $(date +%s) -lt $end ]; do
  out=$(kubectl -n <namespace> exec rp-v031-a -- bash -c "grep -E -m3 '$PAT|Traceback|EngineCore failed|Engine core initialization failed' /results/logs/$TAG.log" 2>/dev/null)
  if [ -n "$out" ]; then echo "$out" | cut -c1-400; exit 0; fi
  sleep 15
done
echo TIMEOUT
