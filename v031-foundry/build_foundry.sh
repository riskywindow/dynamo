#!/bin/bash
# In-pod (vllm-runtime 1.6.0 = vLLM 0.31.0, torch 2.13.0+cu130, aarch64, user dynamo, read-only site-packages):
#  1. build upstream Foundry (3e034b0) from source against the image's torch into the user site (no aarch64 wheels)
#  2. copy the image's vllm package to /results/vllm-site and apply the adapter overlay there (PR #66472 rebased);
#     engines run with PYTHONPATH=/results/vllm-site (see env.sh)
set -x
export PATH=$HOME/.local/bin:$PATH
# the image's /usr/local/cuda lacks some dev headers torch's ATen includes (cusparse.h); torch's pip CUDA bundle has them.
# CPLUS_INCLUDE_PATH is searched after the explicit -I dirs, so it only fills in what /usr/local/cuda/include lacks.
NV=$(python3 -c "import os,nvidia;print(os.path.dirname(nvidia.__path__[0]))")/nvidia/cu13/include
# nvml.h (c10/cuda/driver_api.h) is in neither; take it from NVIDIA's nvidia-nvml-dev wheel (header only at build time).
mkdir -p /results/cuda-extra && [ -f /results/cuda-extra/nvidia/cu13/include/nvml.h ] || {
  pip download -q --no-deps -d /results/cuda-extra/whl "nvidia-nvml-dev==13.*" && \
  python3 -m zipfile -e /results/cuda-extra/whl/nvidia_nvml_dev-*.whl /results/cuda-extra; }
EX=/results/cuda-extra/nvidia/cu13/include
export CPLUS_INCLUDE_PATH=$NV:$EX${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH} C_INCLUDE_PATH=$NV:$EX${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}
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
