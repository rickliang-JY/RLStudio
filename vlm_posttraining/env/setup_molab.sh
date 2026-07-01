#!/usr/bin/env bash
# Phase 0 一键引导(molab / marimo lab):验证环境 + 跑通 Qwen2-VL-2B 推理。
# 用法:  bash env/setup_molab.sh
# 设计原则:molab 已预装 torch 2.12/cu130(Blackwell 原生),绝不重装 torch;
#          molab 在美区,直连 HuggingFace,不用 network_turbo / hf-mirror。

set -uo pipefail

hr() { printf '\n############ %s ############\n' "$1"; }

hr "STEP 0  平台确认"
if [ -d /marimo ]; then
  echo "[ok] 检测到 /marimo(molab 持久盘)"
else
  echo "[warn] 未检测到 /marimo —— 这不是 molab?若在 AutoDL 请改用 setup_autodl.sh"
fi

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
        print(">>> Blackwell(PRO 6000 / 5090)级:molab 预装 torch 已原生支持,勿重装。")
except Exception as e:
    print("!! torch 检查失败:", e)
PY

hr "STEP 2  缓存指向持久盘 /marimo(跨 session 复用,免重下)"
export HF_HOME=/marimo/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

hr "STEP 3  安装推理依赖(不重装 torch)"
TORCH_BEFORE=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
pip install -q -U "transformers>=4.49" accelerate qwen-vl-utils pillow hf_transfer
TORCH_AFTER=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo none)
python -c "import transformers; print('transformers', transformers.__version__)"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "!! 警告:torch 被改动($TORCH_BEFORE -> $TORCH_AFTER)。molab 原生 cu130 可能失效,建议 pip 装回原版或重开机器。"
fi

hr "STEP 4  Phase 0 推理 sanity(走 HuggingFace,美区直连)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../phase0_architecture"
python inference_sanity.py

hr "DONE"
echo "若上面打印出模型回答且提到 red/circle,Phase 0 通过 —— ViT→connector→LLM 全链路 + Blackwell 环境验证完成。"
