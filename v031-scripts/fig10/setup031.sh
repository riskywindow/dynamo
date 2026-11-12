#!/bin/bash
# setup031.sh: GMS server + primary (writer) on :8000, then a matched lean-Foundry SAVE shadow on :8001 (killed after ready).
source /results/f10/f10lib.sh; mkdir -p $F/logs $F/ws
for m in save load; do
  printf 'mode = "%s"\nworkspace_root = "%s"\nbase_addr = "0x600000000000"\nregion_size = "512GB"\nscratch_space_size = "2GB"\nskip_cudagraph_memory_profile = true\nskip_capture_warmup = true\n' $m $F/ws > $F/foundry-$m.toml
  [ -n "$FOUNDRY_TOML_EXTRA" ] && printf '%s\n' "$FOUNDRY_TOML_EXTRA" >> $F/foundry-$m.toml
done
pkill -9 -f "[v]llm serve"; pkill -9 -f "[V]LLM::"; pkill -9 -f "[g]pu_memory_service --device"; gpus_free
bash $S/start_gms.sh
bash $F/serve031.sh primary 8000 $F/logs/primary.log; wait_ready 8000 $F/logs/primary.log 1200 || exit 1
echo "primary ready $(date +%s.%N)"
rm -rf $F/ws; mkdir -p $F/ws
bash $F/serve031.sh shadow 8001 $F/logs/save.log $F/foundry-save.toml; wait_ready 8001 $F/logs/save.log || exit 1
echo "save ready; breakable_saved=$(grep -c 'breakable SAVE' $F/logs/save.log)"
kill_engine $F/logs/save.log; pkill -9 -f "[v]llm serve.*--port 8001"; sleep 5
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '; echo
