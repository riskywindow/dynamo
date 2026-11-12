#!/bin/bash
# model_run.sh NAME MODEL_DIR "engine args" "base LLM kwargs json body" [SIZES]
# Fresh GMS servers, then the full lean path (foundry_shadow.sh with both lean switches).
S=/results/scripts
bash $S/kill_engines.sh all >/dev/null; pkill -9 -f "[g]pu_memory_service --device"
for i in $(seq 60); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1); [ "$u" -lt 2000 ] && break; sleep 2; done
bash $S/start_gms.sh >/dev/null
FOUNDRY_TOML_EXTRA=$'skip_cudagraph_memory_profile = true\nskip_capture_warmup = true' \
  bash $S/foundry_shadow.sh "$1" "$2" "$3" "$4" "${5:-[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]}" 512GB 2>&1 | grep -v "nondeterministic ("
