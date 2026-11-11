#!/bin/bash
# GMS v0 server (patched, one process per GPU) in its own session.
source /results/scripts/env.sh
mkdir -p $GMS_SOCKET_DIR; rm -f $GMS_SOCKET_DIR/gms_*.sock
setsid nohup bash -c "python3 -m gpu_memory_service.cli.server > /results/logs/gms.log 2>&1" < /dev/null > /dev/null 2>&1 &
for i in $(seq 120); do [ "$(ls $GMS_SOCKET_DIR/gms_*.sock 2>/dev/null | wc -l)" -ge 8 ] && break; sleep 1; done
echo "gms sockets: $(ls $GMS_SOCKET_DIR/gms_*.sock 2>/dev/null | wc -l)"
