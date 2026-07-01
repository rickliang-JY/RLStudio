#!/usr/bin/env bash
# 各阶段共享变量与工具,被各 run_*.sh `source`。
# 约定:单卡(5090 / PRO 6000)+ Qwen2-VL-2B,全程 LoRA,产物全放持久数据盘。

# --- 数据盘根目录:自动探测平台 ---
# molab 用 /marimo(7T 持久盘);AutoDL 用 /root/autodl-tmp;都没有则退回 $HOME。
# 想强制指定:export DATA_ROOT=/your/path 后再 source。
if [ -z "${DATA_ROOT:-}" ]; then
  if   [ -d /marimo ];          then DATA_ROOT=/marimo
  elif [ -d /root/autodl-tmp ]; then DATA_ROOT=/root/autodl-tmp
  else                               DATA_ROOT=$HOME/vlm-data
  fi
fi
export DATA_ROOT

# --- 缓存 / 设备 ---
export HF_HOME=${HF_HOME:-$DATA_ROOT/hf_cache}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-$DATA_ROOT/modelscope_cache}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# 下载源统一走 ModelScope(USE_HF=0)。原因:本项目的多模态数据集(coco_2014_caption、
# RLAIF-V 偏好数据)是 ModelScope 独有,且带 ms-swift 注册好的预处理格式;HF 上要么没有、
# 要么列名对不上。molab 在美区拉 ModelScope 稍慢但能通。想强制走 HF:export USE_HF=1。
export USE_HF=${USE_HF:-0}

# --- 路径 ---
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2-VL-2B-Instruct}
export WORK=${WORK:-$DATA_ROOT/vlm}
export CKPT_DIR=$WORK/ckpt        # LoRA 训练输出(adapter)
export MODEL_DIR=$WORK/models     # merge 后的完整模型(供下游 / 评测 / 量化)
export LOG_DIR=$WORK/logs
mkdir -p "$CKPT_DIR" "$MODEL_DIR" "$LOG_DIR"

# --- 训练通用超参(单卡省显存档)---
export REPORT=${REPORT:-tensorboard}   # 想用 wandb:export REPORT=wandb 并先 `wandb login`
export LORA_RANK=${LORA_RANK:-8}
export LORA_ALPHA=${LORA_ALPHA:-32}
export MAX_LEN=${MAX_LEN:-2048}
export GA=${GA:-16}                    # 梯度累积,等效放大 batch

# 取某 output_dir 下「最新」的 checkpoint-*(按修改时间)
last_ckpt() {
  find "$1" -maxdepth 3 -type d -name 'checkpoint-*' -printf '%T@\t%p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -f2-
}

# 把某个 LoRA checkpoint merge 成完整模型,输出到 $MODEL_DIR/<name>
# 用法: merge_lora <checkpoint_dir> <out_name>
merge_lora() {
  local ckpt="$1" name="$2"
  echo ">>> merge LoRA: $ckpt -> $MODEL_DIR/$name"
  swift export --adapters "$ckpt" --merge_lora true --output_dir "$MODEL_DIR/$name"
}

echo "[common] DATA_ROOT=$DATA_ROOT  BASE_MODEL=$BASE_MODEL  WORK=$WORK  GPU=$CUDA_VISIBLE_DEVICES  REPORT=$REPORT"
