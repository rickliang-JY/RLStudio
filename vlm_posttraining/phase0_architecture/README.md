# Phase 0 — 看懂结构(打地基)

**状态:** 🚧 脚本就绪,待在 AutoDL 实例上验证

## 目标
不急着训,先把一个小 VLM 跑起来做推理,亲手看清 ViT / connector / LLM 各自吃进吐出什么。

## 做什么
- 加载 `Qwen2-VL-2B-Instruct`,跑一次图片问答推理(环境 sanity check,见 [`../env/`](../env/))。
- 打印各部件的输入/输出张量形状:图片 → 视觉 token → 投影到词向量空间 → LLM 生成。
- (可选)对照 `Mini-LLaVA` 极简实现,逐行读懂前向流程。

## 对应 JD
模型架构理解(LLaVA/Qwen-VL + ViT + connector)。

## 交付物
- `inference_sanity.py`:5090 上跑通的最小推理脚本。
- 一个 marimo notebook 或 markdown,图解各部件张量流(延续 RLStudio「看得见」风格)。

## 必读
[R2] LLaVA · [R3] LLaVA-1.5 · [R5] CLIP · [R4] ViT · [R9] Qwen2-VL
