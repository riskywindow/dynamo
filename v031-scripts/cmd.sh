#!/bin/bash
# cmd.sh TAG "command" -> waits for ack (up to 30 min)
TAG=$1; CMD=$2
n0=$(cat /results/ctl/$TAG.ack 2>/dev/null | wc -l)
echo "$CMD" > /results/ctl/$TAG.cmd
for i in $(seq 3600); do n=$(cat /results/ctl/$TAG.ack 2>/dev/null | wc -l); [ "$n" -gt "$n0" ] && break; sleep 0.5; done
tail -1 /results/ctl/$TAG.ack
