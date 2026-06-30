#!/usr/bin/env bash
# Phase 0 一键引导(AutoDL):验证环境 + 跑通 Qwen2-VL-2B 推理。
# 用法:  bash setup_autodl.sh
# 设计原则:不动镜像自带的 torch(它和镜像的 CUDA 匹配),只装上层推理依赖。

set -uo pipefail

hr() { printf '\n############ %s ############\n' "$1"; }

hr "STEP 0  网络加速(clone / 下载用)"
if source /etc/network_turbo 2>/dev/null; then echo "[ok] network_turbo 已开启"; else echo "[skip] 无 network_turbo,忽略"; fi

hr "STEP 1  环境信息"
echo "--- GPU ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>/dev/null || nvidia-smi
echo "--- python ---"; python --version
echo "--- torch ---"
python - <<'PY'
try:
    import torch
    print(f"torch {torch.__version__} | cuda {torch.version.cuda} | "
          f"{torch.cuda.get_device_name(0)} | cap sm_{''.join(map(str, torch.cuda.get_device_capability(0)))} | "
          f"bf16 {torch.cuda.is_bf16_supported()}")
    cap = torch.cuda.get_device_capability(0)
    if cap[0] >= 12:
        print(">>> Blackwell(5090)级:需 CUDA 12.8+ / torch 2.7+。上面 torch 能正常打印即说明镜像 OK。")
except Exception as e:
    print("!! torch 检查失败:", e)
    print("!! 若报 'no kernel image',说明镜像 CUDA/torch 太老,需在 AutoDL 换更新的镜像。")
PY

hr "STEP 2  缓存指向数据盘(别塞满系统盘)"
export HF_HOME=/root/autodl-tmp/hf_cache
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
mkdir -p "$HF_HOME" "$MODELSCOPE_CACHE"
echo "HF_HOME=$HF_HOME"
echo "MODELSCOPE_CACHE=$MODELSCOPE_CACHE"

hr "STEP 3  安装推理依赖(不重装 torch)"
pip install -q -U "transformers>=4.49" accelerate "qwen-vl-utils" pillow modelscope
python -c "import transformers; print('transformers', transformers.__version__)"

hr "STEP 4  Phase 0 推理 sanity(走 ModelScope 下载,国内更稳)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../phase0_architecture"
USE_MODELSCOPE=1 python inference_sanity.py

hr "DONE"
echo "若上面打印出模型回答且提到 red/circle,Phase 0 通过 —— ViT→connector→LLM 全链路 + 5090 环境验证完成。"
