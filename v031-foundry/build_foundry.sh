#!/bin/bash
# In-pod (vllm-runtime 1.6.0 = vLLM 0.31.0, torch 2.13.0+cu130, aarch64, user dynamo, read-only site-packages):
#  1. build upstream Foundry (3e034b0) from source against the image's torch into the user site (no aarch64 wheels)
#  2. copy the image's vllm package to /results/vllm-site and apply the adapter overlay there (PR #66472 rebased);
#     engines run with PYTHONPATH=/results/vllm-site (see env.sh)
set -x
export PATH=$HOME/.local/bin:$PATH
cd /results && [ -d foundry ] || tar xzf foundry-main.tgz
pip install -q --user "cmake>=4.0,<5" 2>&1 | tail -1; cmake --version | head -1
cd /results/foundry && MAX_JOBS=32 TORCH_CUDA_ARCH_LIST="10.0" pip install -v --user --no-build-isolation --no-deps . > /results/foundry-build.log 2>&1
echo "foundry_build_rc=$?"; grep -i -E "^error|CMake Error" /results/foundry-build.log | head -5
cd /tmp && python3 -c "import foundry, foundry.ops; print('foundry', getattr(foundry,'__version__','?'), foundry.__file__)"
V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))")
mkdir -p /results/vllm-site && [ -d /results/vllm-site/vllm ] || cp -a $V /results/vllm-site/vllm
cp -a /results/overlay/vllm/. /results/vllm-site/vllm/
cd /tmp && PYTHONPATH=/results/vllm-site python3 -c "import vllm, vllm.compilation.foundry_adapter as a; from vllm.config import CompilationConfig; print('adapter ok', vllm.__file__, 'graph_extension_config_path' in CompilationConfig.__dataclass_fields__)"
echo BUILD_DONE
