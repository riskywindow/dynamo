# f10lib.sh: helpers (source me)
F=/results/f10; S=/results/scripts
ready() { curl -sf localhost:$1/v1/models > /dev/null; }
wait_ready() {  # PORT LOG [max_s]
  for i in $(seq ${3:-900}); do ready $1 && return 0
    grep -qE "Traceback|Segfault|EngineCore failed|Engine core initialization failed" $2 2>/dev/null && { echo "DIED $2"; grep -E "Error" $2 | tail -5 | cut -c1-300; return 1; }
    sleep 1; done; echo "TIMEOUT $2"; return 1; }
kill_engine() { [ -f $1.pgid ] && kill -9 -- -$(cat $1.pgid) 2>/dev/null; sleep 2; }
gpus_free() { for i in $(seq 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 2000 ] && return; sleep 2; done; }
