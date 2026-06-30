# Phase 6 — 评测(把每一步的效果量化)

**状态:** ⬜ 未开始

## 概念
有了 base/SFT/DPO/GRPO 几个 checkpoint,客观比较:
- **幻觉降了多少**:POPE / MMHal-Bench / CHAIR。
- **通用能力掉了没**(alignment tax):MM-Vet / MMBench。

## 做什么
- 用 `VLMEvalKit` 一键扫全部 checkpoint。

## 对应 JD
系统评测、熟悉顶会 benchmark、实验管理。

## 交付物
- 统一评测脚本 + 一张跨 checkpoint 的结果对比表(进技术报告的核心图表)。

## 必读
[R30] VLMEvalKit · [R31] POPE · [R32] CHAIR · [R33] MMHal-Bench · [R34] MM-Vet · [R35] MMBench
