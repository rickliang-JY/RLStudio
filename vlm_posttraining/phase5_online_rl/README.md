# Phase 5 — 在线 RL(PPO / GRPO)★ JD 加分项

**状态:** ⬜ 未开始

## 概念
模型生成回答 → RM 打分 → 更新模型。PPO 经典;GRPO 更省(去掉 value network)。这是「在线 RLHF」。

## 做什么
- 用 Phase 4 的 RM,跑一轮 GRPO(更易上手)或 PPO。
- 单卡注意:用 vLLM colocate 加速生成;7B 偏紧,主线先在 2B 上跑。
- 深入 infra 可参考 verl / OpenRLHF。

## 对应 JD
后训练对齐(PPO 在线 RLHF)。

## 交付物
- GRPO/PPO 配置 + 启动脚本 + reward 曲线(W&B)。
- checkpoint:`qwen2vl2b-grpo`。

## 必读
[R17] PPO · [R18] GRPO(DeepSeekMath)· [R19] DeepSeek-R1
