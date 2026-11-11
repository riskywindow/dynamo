export PYTHONPATH=/home/dynamo/gms-v031${PYTHONPATH:+:$PYTHONPATH}
export GMS_SOCKET_DIR=/tmp/gms
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p /results/logs /results/out /results/ctl
