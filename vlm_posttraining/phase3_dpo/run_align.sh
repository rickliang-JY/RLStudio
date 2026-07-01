#!/usr/bin/env bash
# Phase 3 — 离线偏好对齐(LoRA),ms-swift `swift rlhf`。
# 用法:  bash run_align.sh dpo      # 或 orpo / kto
# 基座:Phase 1 的 SFT 模型($MODEL_DIR/qwen2vl2b-sft);没跑 SFT 就回退到 $BASE_MODEL。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../env/common.sh"

TYPE=${1:-dpo}                                   # dpo | orpo | kto
SFT_MODEL=${SFT_MODEL:-$MODEL_DIR/qwen2vl2b-sft}
[ -d "$SFT_MODEL" ] || { echo "!! 未找到 SFT 模型,回退到 $BASE_MODEL"; SFT_MODEL=$BASE_MODEL; }
DATASET=${DATASET:-'swift/RLAIF-V-Dataset#5000'}

# KTO 用 prospect-theory 损失,数据形态更灵活;ms-swift 可直接吃成对偏好数据。
echo ">>> $TYPE align on $DATASET  (base: $SFT_MODEL)"
swift rlhf \
  --rlhf_type "$TYPE" \
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
  --warmup_ratio 0.03 \
  --save_steps 200 --save_total_limit 2 \
  --logging_steps 5 \
  --report_to "$REPORT" \
  --output_dir "$CKPT_DIR/$TYPE" 2>&1 | tee "$LOG_DIR/$TYPE.log"
st=${PIPESTATUS[0]}; [ "$st" -eq 0 ] || { echo "!! $TYPE 训练失败(exit $st),见上方报错 / $LOG_DIR/$TYPE.log"; exit "$st"; }

CKPT=$(last_ckpt "$CKPT_DIR/$TYPE")
[ -n "$CKPT" ] && merge_lora "$CKPT" "qwen2vl2b-$TYPE" || echo "!! 未找到 $TYPE checkpoint"
echo ">>> $TYPE 完成。三种都跑: for t in dpo orpo kto; do bash run_align.sh \$t; done"
