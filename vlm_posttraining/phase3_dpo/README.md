# Phase 3 — DPO 家族(离线偏好对齐)★ JD 核心

**状态:** ⬜ 未开始

## 概念
DPO 直接让模型「偏向好回答、远离差回答」,不需单独训奖励模型。
同类有 ORPO、KTO、SimPO,做法略不同。

## 做什么
- 先 DPO,再各跑一版 ORPO / KTO,横向比较效果。
- 命令:`swift rlhf --rlhf_type dpo ...`。

## 对应 JD
后训练对齐(DPO/ORPO/KTO)——岗位职责明文核心。

## 交付物
- 各方法的训练配置 + 启动脚本。
- checkpoint:`qwen2vl2b-dpo` / `-orpo` / `-kto`,交 Phase 6 统一评测。

## 必读
[R13] DPO · [R14] ORPO · [R15] KTO · [R16] SimPO
