#!/usr/bin/env bash
# Phase 4 — 奖励模型(RM)训练,ms-swift `swift rlhf --rlhf_type rm`。
# 在 SFT 模型上加一个打分头,学「好回答 > 差回答」。产出供 Phase 5 在线 RL 当 reward。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../env/common.sh"

SFT_MODEL=${SFT_MODEL:-$MODEL_DIR/qwen2vl2b-sft}
[ -d "$SFT_MODEL" ] || { echo "!! 未找到 SFT 模型,回退到 $BASE_MODEL"; SFT_MODEL=$BASE_MODEL; }
DATASET=${DATASET:-'swift/RLAIF-V-Dataset#5000'}

echo ">>> RM on $DATASET  (base: $SFT_MODEL)"
swift rlhf \
  --rlhf_type rm \
  --model "$SFT_MODEL" \
  --dataset "$DATASET" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GA" \
  --learning_rate 5e-5 \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" \
  --target_modules all-linear \
  --freeze_vit true \
  --gradient_checkpointing true \
  --max_length "$MAX_LEN" \
  --save_steps 200 --save_total_limit 2 \
  --logging_steps 5 \
  --report_to "$REPORT" \
  --output_dir "$CKPT_DIR/rm" 2>&1 | tee "$LOG_DIR/rm.log"
st=${PIPESTATUS[0]}; [ "$st" -eq 0 ] || { echo "!! RM 训练失败(exit $st),见上方报错 / $LOG_DIR/rm.log"; exit "$st"; }

RM_CKPT=$(last_ckpt "$CKPT_DIR/rm")
[ -n "$RM_CKPT" ] && merge_lora "$RM_CKPT" "qwen2vl2b-rm" || echo "!! 未找到 RM checkpoint"
echo ">>> Phase 4 完成。验证点:训练日志里 eval 的 reward/accuracy(好回答打分高于差回答的比例)。"
