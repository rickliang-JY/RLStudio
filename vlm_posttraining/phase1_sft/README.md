# Phase 1 — SFT(视觉指令微调)

**状态:** 🚧 脚本就绪(`run_sft.sh`),待运行

## 概念
拿「(图,问,标准答案)」数据,教模型看图听话回答。后训练第一关。

## 做什么
- 用 LoRA 在一个 VQA 数据集上做 SFT。
- 先用 `lmms-finetune` 学透原理(轻量透明),再切 `ms-swift` 做生产化复现。

## 对应 JD
微调框架(PEFT/DeepSpeed/Swift)、PEFT/LoRA、多模态数据处理。

## 交付物
- ms-swift 的 SFT 配置 + 启动脚本。
- LoRA 训练曲线(W&B),与 base 的简单对比。
- checkpoint 命名:`qwen2vl2b-sft-lora`。

## 必读
[R10] LoRA · [R11] QLoRA · [R2] LLaVA(两阶段训练范式)
