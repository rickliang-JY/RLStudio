#!/usr/bin/env bash
# Phase 7 — 量化 + 部署。把最好的模型(默认 DPO)量化成 AWQ/GPTQ,再用 vLLM 部署。
# 用法:  bash run_quant_deploy.sh quant    # 仅导出量化模型
#         bash run_quant_deploy.sh deploy   # 起 vLLM 服务
#         bash run_quant_deploy.sh bench    # 简单吞吐/延迟 benchmark
# ⚠ 5090(sm_120):AWQ/GPTQ kernel 需较新版本,报 kernel 错就升级 autoawq / optimum / vllm。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../env/common.sh"

ACTION=${1:-quant}
SRC=${SRC:-$MODEL_DIR/qwen2vl2b-dpo}            # 要量化/部署的源模型
[ -d "$SRC" ] || { echo "!! 源模型不存在: $SRC(先跑对应阶段或改 SRC=)"; exit 1; }
METHOD=${METHOD:-awq}                            # awq | gptq
QMODEL=$MODEL_DIR/$(basename "$SRC")-$METHOD
CALIB=${CALIB:-'modelscope/coco_2014_caption#256'}   # 量化校准数据
PORT=${PORT:-8000}

case "$ACTION" in
  quant)
    echo ">>> $METHOD 量化: $SRC -> $QMODEL"
    swift export \
      --model "$SRC" \
      --quant_method "$METHOD" --quant_bits 4 \
      --dataset "$CALIB" \
      --output_dir "$QMODEL" 2>&1 | tee "$LOG_DIR/quant_$METHOD.log"
    echo ">>> 量化完成。对比量化前后体积: du -sh $SRC $QMODEL"
    ;;
  deploy)
    echo ">>> vLLM 部署 $QMODEL  (port $PORT)"
    swift deploy --model "$QMODEL" --infer_backend vllm --port "$PORT" \
      2>&1 | tee "$LOG_DIR/deploy_$METHOD.log"
    ;;
  bench)
    echo ">>> 简单 benchmark(需先 deploy 起服务)"
    python - "$PORT" <<'PY'
import sys, time, requests
port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
payload = {"model": "default", "messages": [{"role": "user", "content": "用一句话介绍长城。"}], "max_tokens": 128}
t0 = time.time()
r = requests.post(url, json=payload, timeout=60)
dt = time.time() - t0
print("status", r.status_code, "| 耗时 %.2fs" % dt)
print(r.json().get("choices", [{}])[0].get("message", {}).get("content", r.text)[:300])
PY
    ;;
  *) echo "未知动作: $ACTION (quant|deploy|bench)";;
esac
echo ">>> Phase 7:量化前后用 Phase 6 再评一遍,记录 精度↓ vs 体积↓ vs 速度↑。"
