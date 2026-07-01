#!/usr/bin/env bash
# 安装训练/评测/部署框架(Phase 1 起)。关键:别让 pip 把镜像自带、与 CUDA 匹配的 torch 换掉。
# 用法:  bash setup_swift.sh            # 装 ms-swift(SFT/DPO/RM 够用)
#         bash setup_swift.sh full      # 额外装 vllm + evalscope(Phase 5/6/7 需要)
set -uo pipefail
source /etc/network_turbo 2>/dev/null || true

TORCH_BEFORE=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo "none")
echo "[before] torch=$TORCH_BEFORE"

echo ">>> 安装 ms-swift"
pip install -q -U ms-swift

# datasets 4.0 起彻底删掉了「加载脚本」支持(会报 "Dataset scripts are no longer supported"),
# 而 ms-swift 的多模态数据集(coco_2014_caption 等)不少仍以脚本形式托管。molab 预装的是 datasets 4.x,
# 直接跑会在加载数据这步崩。ms-swift 4.3.2 允许 datasets>=3.0,<4.8.5,所以退到 3.x 是官方支持的合法配置,
# 且能一次性修好所有阶段的脚本型数据集。只动 datasets,不碰 torch。
echo ">>> 固定 datasets<4(恢复脚本型数据集支持)"
pip install -q "datasets<4"
python -c "import datasets; print('[datasets]', datasets.__version__)"

if [ "${1:-}" = "full" ]; then
  echo ">>> 安装 vllm + evalscope(VLMEvalKit 后端)"
  # 注:vllm 版本需与 torch/CUDA 匹配;5090 需较新 vllm。若它想换 torch,先 Ctrl-C 改用 --no-deps 手动处理。
  pip install -q -U vllm "evalscope[vlmeval]" qwen-vl-utils
fi

TORCH_AFTER=$(python -c "import torch;print(torch.__version__)" 2>/dev/null || echo "none")
echo "[after]  torch=$TORCH_AFTER"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "!! 警告:torch 版本被改了($TORCH_BEFORE -> $TORCH_AFTER)。5090 上若变成非 cu128 版可能跑不了 kernel。"
  echo "!! 建议:卸掉再装回镜像匹配版,或重开实例。验证:python -c 'import torch;print(torch.cuda.get_device_capability(0))'"
fi
swift --version 2>/dev/null || python -c "import swift;print('ms-swift', swift.__version__)"
echo ">>> 框架安装完成。"
