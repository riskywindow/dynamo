# source in-pod before run_arm.sh: patched vLLM copy first, user site (Foundry) and user bin on PATH
export PYTHONPATH=/results/vllm-site${PYTHONPATH:+:$PYTHONPATH} PATH=$HOME/.local/bin:$PATH
