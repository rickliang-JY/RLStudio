#!/usr/bin/env bash
# Phase 1 — 视觉指令微调(LoRA SFT),ms-swift。
# 产出: $MODEL_DIR/qwen2vl2b-sft (merge 后完整模型),供 Phase 3/4 当基座。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../env/common.sh"

# 数据集:ms-swift 内置多模态数据,按名即用。下面默认用 COCO caption 子集做 SFT 演示。
# 想换 VQA 类:export DATASET='swift/OK-VQA_train#5000' 等(见 ms-swift 文档「支持的数据集」)。
DATASET=${DATASET:-'modelscope/coco_2014_caption#3000'}

echo ">>> SFT on $DATASET"
swift sft \
  --model "$BASE_MODEL" \
  --dataset "$DATASET" \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GA" \
  --learning_rate 1e-4 \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" \
  --target_modules all-linear \
  --freeze_vit true \
  --gradient_checkpointing true \
  --max_length "$MAX_LEN" \
  --warmup_ratio 0.03 \
  --eval_steps 200 --save_steps 200 --save_total_limit 2 \
  --logging_steps 5 \
  --dataloader_num_workers 4 \
  --report_to "$REPORT" \
  --output_dir "$CKPT_DIR/sft" 2>&1 | tee "$LOG_DIR/sft.log"
st=${PIPESTATUS[0]}; [ "$st" -eq 0 ] || { echo "!! SFT 训练失败(exit $st),见上方报错 / $LOG_DIR/sft.log"; exit "$st"; }

# merge 成完整模型
SFT_CKPT=$(last_ckpt "$CKPT_DIR/sft")
[ -n "$SFT_CKPT" ] && merge_lora "$SFT_CKPT" "qwen2vl2b-sft" || echo "!! 未找到 SFT checkpoint"
echo ">>> Phase 1 完成。下一步: bash ../phase3_dpo/run_align.sh dpo"
