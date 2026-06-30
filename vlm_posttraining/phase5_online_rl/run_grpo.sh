#!/usr/bin/env bash
# Phase 5 — 在线 RL:GRPO(去掉 value network 的 PPO 简化版),ms-swift。
# ★ 全流程最吃资源/最容易要调的一步。单卡 5090 + 2B 用 vLLM colocate 加速采样。
# 基座:Phase 1 SFT 模型;reward:Phase 4 的 RM。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "$HERE/../env/common.sh"

SFT_MODEL=${SFT_MODEL:-$MODEL_DIR/qwen2vl2b-sft}
RM_MODEL=${RM_MODEL:-$MODEL_DIR/qwen2vl2b-rm}
[ -d "$SFT_MODEL" ] || { echo "!! 未找到 SFT 模型,回退 $BASE_MODEL"; SFT_MODEL=$BASE_MODEL; }
[ -d "$RM_MODEL" ]  || echo "!! 未找到 RM($RM_MODEL),需先跑 Phase 4;或改用 --reward_funcs 规则奖励。"
DATASET=${DATASET:-'swift/RLAIF-V-Dataset#2000'}
NGEN=${NGEN:-4}              # 每个 prompt 采样数(组内比较);显存紧就调小

echo ">>> GRPO  base=$SFT_MODEL  rm=$RM_MODEL  ngen=$NGEN"
swift rlhf \
  --rlhf_type grpo \
  --model "$SFT_MODEL" \
  --reward_model "$RM_MODEL" \
  --dataset "$DATASET" \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_generations "$NGEN" \
  --use_vllm true --vllm_mode colocate --vllm_gpu_memory_utilization 0.4 \
  --max_completion_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GA" \
  --learning_rate 1e-6 \
  --lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA" \
  --target_modules all-linear \
  --freeze_vit true \
  --gradient_checkpointing true \
  --max_length "$MAX_LEN" \
  --save_steps 100 --save_total_limit 2 \
  --logging_steps 2 \
  --report_to "$REPORT" \
  --output_dir "$CKPT_DIR/grpo" 2>&1 | tee "$LOG_DIR/grpo.log"

CKPT=$(last_ckpt "$CKPT_DIR/grpo")
[ -n "$CKPT" ] && merge_lora "$CKPT" "qwen2vl2b-grpo" || echo "!! 未找到 GRPO checkpoint"
echo ">>> Phase 5 完成。看 reward 曲线是否上升(tensorboard/wandb)。OOM 就调小 NGEN / vllm_gpu_memory_utilization / max_completion_length。"
