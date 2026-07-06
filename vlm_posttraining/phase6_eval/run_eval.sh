#!/usr/bin/env bash
# Phase 6 — 评测。用 ms-swift 的 `swift eval`(底层 EvalScope/VLMEvalKit)横扫各阶段 checkpoint。
# 对比两类指标:幻觉(POPE / MMHal)+ 通用能力(MMBench / MM-Vet,看 alignment tax)。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../env/common.sh"

# 要评测的模型:基座 + 各阶段 merge 后的完整模型(存在哪个评哪个)。
MODELS=(
  "$BASE_MODEL"
  "$MODEL_DIR/qwen2vl2b-sft"
  "$MODEL_DIR/qwen2vl2b-dpo"
  "$MODEL_DIR/qwen2vl2b-orpo"
  "$MODEL_DIR/qwen2vl2b-kto"
  "$MODEL_DIR/qwen2vl2b-grpo"
)
DATASETS=${DATASETS:-'POPE MMBench_DEV_EN MMVet'}
OUT=$WORK/eval_results; mkdir -p "$OUT"

for m in "${MODELS[@]}"; do
  # 基座是仓库名(如 Qwen/Qwen2-VL-2B-Instruct,非绝对路径),其余是本地目录、要求存在
  if [[ "$m" != /* || -d "$m" ]]; then
    tag=$(basename "$m")
    echo "==================== EVAL: $tag ===================="
    swift eval \
      --model "$m" \
      --infer_backend vllm \
      --eval_dataset $DATASETS \
      --eval_output_dir "$OUT/$tag" 2>&1 | tee "$LOG_DIR/eval_$tag.log"
  else
    echo "[skip] 不存在: $m"
  fi
done
echo ">>> Phase 6 完成。把各模型在 POPE/MMBench/MMVet 的分数汇成一张对比表 → 进技术报告。"
echo ">>> 备选:也可直接用 VLMEvalKit 仓库(github.com/open-compass/VLMEvalKit)自定义更多 benchmark。"
