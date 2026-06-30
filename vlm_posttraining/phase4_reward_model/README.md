# Phase 4 — 奖励模型(Reward Model)

**状态:** 🚧 脚本就绪(`run_rm.sh`),待运行

## 概念
训一个「打分员」模型,给任意回答打质量分。在线 RL 的前置零件。

## 做什么
- 在偏好数据上训 RM,验证它能否把好回答打高分。
- ms-swift 原生支持 RM 训练。

## 对应 JD
奖励模型训练。

## 交付物
- RM 训练脚本 + 在留出集上的打分准确率(好回答 > 差回答 的比例)。
- checkpoint:`qwen2vl2b-rm`。

## 必读
[R12] InstructGPT(RLHF 范式与 RM 训练)· [R22] MM-RLHF(多模态 RM)
