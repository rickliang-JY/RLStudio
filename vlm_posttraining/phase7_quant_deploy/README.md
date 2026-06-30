# Phase 7 — 量化与部署

**状态:** ⬜ 未开始

## 概念
把模型压小(AWQ/GPTQ),再用 vLLM 高速部署,测压缩后掉多少点、快多少。

## 做什么
- `ms-swift` 直接导出量化模型(AWQ/GPTQ/FP8/BNB)。
- vLLM 部署并 benchmark(吞吐 / 延迟 / 显存)。
- 注意:5090(sm_120)上量化 kernel 需较新版本,见 [`../env/`](../env/)。

## 对应 JD
量化(GPTQ/AWQ/bitsandbytes)与部署(vLLM/TGI)。

## 交付物
- 量化导出脚本 + 量化前后 benchmark 对比(精度 vs 速度 vs 显存)。

## 必读
[R40] GPTQ · [R41] AWQ · [R42] vLLM/PagedAttention
