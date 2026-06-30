# 多模态大模型后训练 · 技术档案与论文清单

> 用途:为「VLM 后训练全流程复现」项目提供一份可执行的技术地图 + 一份按主题分类的必读论文/仓库清单。
> 目标岗位:北大深圳研究生院 · 多模态大模型后训练科研助理。
> 项目定位:**不追求创新,追求全面与扎实**——把一个小 VLM 的后训练全生命周期从头到尾走通,JD 技能表逐项打卡。

---

## 1. 项目总览

**一句话:** 拿一个小 VLM(推荐 Qwen2-VL-2B,有余力再上 7B),把它的后训练全流程完整跑一遍——从看懂结构,到 SFT → 偏好数据 → DPO 家族 → 奖励模型 → PPO/GRPO → 评测 → 量化部署,每一站都站在成熟开源仓库上。

**最终交付物(= JD 申请材料):**
- 一个结构清晰的 GitHub 仓库(每个 Phase 一个目录 + README)
- 一份技术报告(把 base / SFT / DPO / GRPO 各阶段的 benchmark 数字列表对比,讲清每一步带来什么变化)

**主框架:** `ms-swift`(一个框架覆盖 SFT/DPO/GRPO/PPO/RM + 量化 + 部署 + 评测,且 Swift 本身就在 JD 点名的框架里)。
**评测:** `VLMEvalKit`(一条命令跑 80+ benchmark)。
**偏好数据:** `RLAIF-V` / `RLHF-V`(现成,不用自己标注)。

---

## 2. VLM 架构心智模型

一个 VLM(以 LLaVA 为代表)由三块拼成:

```
图片 ──► [ViT 视觉编码器] ──► 视觉 token ──► [Connector / MLP] ──► LLM 词向量空间
                                                                      │
文本 ──────────────────────────────► 文本 token ─────────────────────┤
                                                                      ▼
                                                                 [LLM] ──► 回答
```

- **ViT(视觉编码器):** 通常是 CLIP ViT-L/14,把图片切成 patch 编码成一串视觉 token(LLaVA-1.5 在 336×336 分辨率下产生 576 个视觉 token)。
- **Connector(连接器 / projector):** 一个 2 层 MLP,把视觉 token「翻译」到 LLM 能理解的词向量空间。这是训练时唯一从零学的部件。
- **LLM:** 接收图文混合 token,生成文本回答。

「后训练」= 基座训好之后、你在它之上做的所有调教(SFT + 对齐 + 压缩部署)。本项目专做这后半段。

---

## 3. 后训练全流程(8 个阶段)

> 关键认知:8 个阶段是**渐进**的,每做完一个就多一项能写进简历的东西。哪怕只做到 Phase 3,也已覆盖 JD 核心。不用担心做不完。

### Phase 0 — 看懂结构(打地基)
- **做什么:** 不急着训,先把一个小 VLM 跑起来做推理,亲手看清 ViT / connector / LLM 各自吃进吐出什么。
- **站在哪:** `Mini-LLaVA`(极简实现,单卡 notebook,小到能读懂每一行)。
- **对应 JD:** 模型架构理解。
- **必读:** [R2] LLaVA · [R3] LLaVA-1.5 · [R5] CLIP · [R4] ViT

### Phase 1 — SFT(视觉指令微调)
- **概念:** 拿「(图,问,标准答案)」数据,教模型看图听话回答。后训练第一关。
- **做什么:** 用 LoRA 在 VQA 数据集上做 SFT。
- **站在哪:** 先用 `lmms-finetune` 学透原理(轻量透明),再切 `ms-swift` 做生产。
- **对应 JD:** 微调框架、PEFT/LoRA、多模态数据处理。
- **必读:** [R10] LoRA · [R11] QLoRA · [R2] LLaVA(两阶段训练范式)

### Phase 2 — 偏好数据(对齐的原料)
- **概念:** 对同一问题,准备「好回答 vs 差回答」的成对数据。后续所有对齐方法靠它喂。
- **做什么:** 用现成的 `RLAIF-V`(83K pairs);进阶跑它的 data generation 脚本,体验偏好数据怎么造出来(采样多个回答→自动判优劣)。
- **对应 JD:** 多模态数据处理、后训练数据集构建。
- **必读:** [R20] RLHF-V · [R21] RLAIF-V

### Phase 3 — DPO 家族(离线偏好对齐)★ JD 核心
- **概念:** DPO 直接让模型「偏向好回答、远离差回答」,不需单独训奖励模型。同类有 ORPO、KTO、SimPO,做法略不同。
- **做什么:** 先 DPO,再各跑一版 ORPO / KTO,比较效果。
- **站在哪:** `ms-swift`(`swift rlhf --rlhf_type dpo ...`)。
- **对应 JD:** 后训练对齐(DPO/ORPO/KTO)——岗位职责明文核心。
- **必读:** [R13] DPO · [R14] ORPO · [R15] KTO · [R16] SimPO

### Phase 4 — 奖励模型(Reward Model)
- **概念:** 训一个「打分员」模型,给任意回答打质量分。在线 RL 的前置零件。
- **做什么:** 在偏好数据上训 RM,验证它能否把好回答打高分。
- **站在哪:** `ms-swift` 原生支持 RM。
- **对应 JD:** 奖励模型训练。
- **必读:** [R12] InstructGPT(RLHF 范式与 RM 训练)· [R22] MM-RLHF(多模态 RM)

### Phase 5 — 在线 RL(PPO / GRPO)★ JD 加分项
- **概念:** 模型生成回答 → RM 打分 → 更新模型。PPO 经典;GRPO 更新更省(去掉 value network)。这是「在线 RLHF」。
- **做什么:** 用 Phase 4 的 RM,跑一轮 GRPO(更易上手)或 PPO。
- **站在哪:** `ms-swift` 的 GRPO 算法家族;深入 infra 看 verl / OpenRLHF。
- **对应 JD:** 后训练对齐(PPO 在线 RLHF)。
- **必读:** [R17] PPO · [R18] GRPO(DeepSeekMath)· [R19] DeepSeek-R1

### Phase 6 — 评测(把每一步的效果量化)
- **概念:** 你有了 base/SFT/DPO/GRPO 几个 checkpoint,要客观比较:幻觉降了多少(POPE/MMHal/CHAIR),通用能力掉了没(MM-Vet/MMBench = alignment tax)。
- **做什么:** 用 `VLMEvalKit` 一键扫全部 checkpoint。
- **对应 JD:** 系统评测、熟悉顶会 benchmark、实验管理。
- **必读:** [R30] VLMEvalKit · [R31] POPE · [R32] CHAIR · [R33] MMHal-Bench · [R34] MM-Vet · [R35] MMBench

### Phase 7 — 量化与部署
- **概念:** 把模型压小(AWQ/GPTQ)、再用 vLLM 高速部署,测压缩后掉多少点、快多少。
- **做什么:** `ms-swift` 直接导出量化模型(AWQ/GPTQ/FP8/BNB),vLLM 部署 benchmark。
- **对应 JD:** 量化(GPTQ/AWQ/bitsandbytes)与部署(vLLM/TGI)。
- **必读:** [R40] GPTQ · [R41] AWQ · [R42] vLLM/PagedAttention

### 贯穿全程 — 工程管理
每阶段都用 **W&B** 记录、**DeepSpeed ZeRO + torchrun** 多卡、跑在 **SLURM** 上。
- **对应 JD:** Python/PyTorch/分布式训练、实验管理。
- **必读:** [R43] DeepSpeed ZeRO

---

## 4. JD 技术栈覆盖映射

| JD 要求 | 本项目对应 Phase |
|---|---|
| 模型架构(LLaVA/Qwen-VL + ViT + connector) | Phase 0 |
| 微调框架(PEFT/DeepSpeed/Swift) | Phase 1(主框架 ms-swift) |
| 多模态数据处理 | Phase 1–2 |
| 高效微调(LoRA/QLoRA) | Phase 1,贯穿 |
| 后训练对齐(DPO/ORPO/KTO + PPO) | Phase 3 + Phase 5 |
| 奖励模型训练 | Phase 4 |
| 量化与部署(GPTQ/AWQ + vLLM) | Phase 7 |
| 工程(PyTorch/分布式/W&B) | 全程 |
| 加分:VLM 幻觉缓解 | Phase 6 主题 |

8 项要求,本项目实打实覆盖 8 项。

---

## 5. 选型与算力建议

- **基座模型:** 首选 **Qwen2-VL-2B-Instruct**(LoRA 全流程在多卡环境完全可行,PPO 也跑得动)。打通后有余力再上 **Qwen2.5-VL-7B** 重做一遍当加强版。
- **主框架锁定 `ms-swift`:** 它一个框架覆盖 SFT/DPO/GRPO/PPO/RM + 量化导出(AWQ/GPTQ/FP8/BNB)+ vLLM/SGLang 部署 + EvalScope 评测。学透一个,JD 大半要求拿下。
- **辅助工具:** `lmms-finetune`(Phase 1 学原理)、`VLMEvalKit`(Phase 6 评测)。
- **务实提醒:** 项目的「难」不在创新,而在把这么长一条链每一环都跑通跑对——环境配置、数据格式、显存调优、checkpoint 衔接,坑不少。能独立把全流程跑通,本身就是科研助理岗最想要的信号。

---

## 6. 完整论文 & 仓库 Reference

> 链接格式:arXiv 论文用 `https://arxiv.org/abs/<ID>`;GitHub 仓库直接给地址。

### 6.1 综述 / 入门(建议最先读)
- **[R1] A Survey of SOTA Large Vision Language Models: Alignment, Benchmark, Evaluations and Challenges** — arXiv:2501.02189 — https://arxiv.org/abs/2501.02189
  全景综述,把 VLM 的对齐方法、benchmark、评测、挑战串成一张图,读完对整个领域有框架感。

### 6.2 VLM 架构基础
- **[R2] Visual Instruction Tuning (LLaVA)** — Liu et al., NeurIPS'23 — arXiv:2304.08485 — https://arxiv.org/abs/2304.08485
  VLM 的奠基作,定义了「ViT + projector + LLM」+ 两阶段训练范式。**必读第一篇。**
- **[R3] Improved Baselines with Visual Instruction Tuning (LLaVA-1.5)** — Liu et al. — arXiv:2310.03744 — https://arxiv.org/abs/2310.03744
  把 connector 换成 2 层 MLP、加 academic VQA 数据,是现在最常用的复现基线。
- **[R4] An Image is Worth 16x16 Words (ViT)** — Dosovitskiy et al., ICLR'21 — arXiv:2010.11929 — https://arxiv.org/abs/2010.11929
  视觉编码器的底层结构。
- **[R5] Learning Transferable Visual Models From Natural Language Supervision (CLIP)** — Radford et al., 2021 — arXiv:2103.00020 — https://arxiv.org/abs/2103.00020
  VLM 里 ViT 编码器的来源,理解视觉-语言对齐的起点。
- **[R6] BLIP-2 (Q-Former)** — Li et al. — arXiv:2301.12597 — https://arxiv.org/abs/2301.12597
  另一种连接器思路(Q-Former),JD 点名要懂。
- **[R7] Flamingo** — Alayrac et al., NeurIPS'22 — arXiv:2204.14198 — https://arxiv.org/abs/2204.14198
  交错图文 + few-shot 多模态的早期代表,JD 点名。
- **[R8] InstructBLIP** — Dai et al. — arXiv:2305.06500 — https://arxiv.org/abs/2305.06500
  指令微调版 BLIP-2,补全架构谱系。
- **[R9] Qwen2-VL** — Wang et al., 2024 — arXiv:2409.12191 — https://arxiv.org/abs/2409.12191
  本项目推荐基座的技术报告,务必读。
- **[R9b] Qwen2.5-VL Technical Report** — Qwen Team, 2025 — arXiv:2502.13923 — https://arxiv.org/abs/2502.13923
  7B 加强版基座的报告。

### 6.3 高效微调 PEFT
- **[R10] LoRA: Low-Rank Adaptation of Large Language Models** — Hu et al., ICLR'22 — arXiv:2106.09685 — https://arxiv.org/abs/2106.09685
  高效微调的基石,**必读**。
- **[R11] QLoRA: Efficient Finetuning of Quantized LLMs** — Dettmers et al., NeurIPS'23 — arXiv:2305.14314 — https://arxiv.org/abs/2305.14314
  4-bit 量化 + LoRA,显存受限时的关键武器。

### 6.4 对齐算法(RLHF / DPO 家族 / GRPO)
- **[R12] Training LMs to Follow Instructions with Human Feedback (InstructGPT)** — Ouyang et al., 2022 — arXiv:2203.02155 — https://arxiv.org/abs/2203.02155
  RLHF 三步范式(SFT → RM → PPO)的源头,**理解对齐必读**。
- **[R13] Direct Preference Optimization (DPO)** — Rafailov et al., NeurIPS'23 — arXiv:2305.18290 — https://arxiv.org/abs/2305.18290
  绕过显式 RM 直接做偏好优化,当前主流起手式,**必读**。
- **[R14] ORPO: Monolithic Preference Optimization without Reference Model** — Hong et al., 2024 — arXiv:2403.07691 — https://arxiv.org/abs/2403.07691
  把 SFT 和偏好优化合成一步,免 reference model。
- **[R15] KTO: Model Alignment as Prospect Theoretic Optimization** — Ethayarajh et al., 2024 — arXiv:2402.01306 — https://arxiv.org/abs/2402.01306
  能从「非成对」数据学习,数据形态更灵活。
- **[R16] SimPO: Simple Preference Optimization with a Reference-Free Reward** — Meng et al., NeurIPS'24 — arXiv:2405.14734 — https://arxiv.org/abs/2405.14734
  用平均 log 概率做隐式 reward,免 reference、更省显存。
- **[R17] Proximal Policy Optimization (PPO)** — Schulman et al., 2017 — arXiv:1707.06347 — https://arxiv.org/abs/1707.06347
  在线 RLHF 的经典策略优化算法,**必读**。
- **[R18] DeepSeekMath (GRPO 出处)** — Shao et al., 2024 — arXiv:2402.03300 — https://arxiv.org/abs/2402.03300
  GRPO 算法的原始论文(去掉 value network 的 PPO 简化版),**必读**。
- **[R19] DeepSeek-R1** — DeepSeek-AI, 2025 — arXiv:2501.12948 — https://arxiv.org/abs/2501.12948
  GRPO 大规模应用、纯 RL 激发推理能力的代表作。

### 6.5 多模态对齐与幻觉
- **[R20] RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback** — Yu et al., CVPR'24 — arXiv:2312.00849 — https://arxiv.org/abs/2312.00849
  段级人工纠错 + DPO 降幻觉,数据效率极高(1.4K 数据 ~35% 降幅)。
- **[R21] RLAIF-V: Open-Source AI Feedback Leads to Super GPT-4V Trustworthiness** — Yu et al., CVPR'25 highlight — arXiv:2405.17220 — https://arxiv.org/abs/2405.17220
  开源模型自动造偏好数据(83K pairs),**本项目 Phase 2 直接用**。
- **[R22] MM-RLHF: The Next Step Forward in Multimodal LLM Alignment** — Zhang et al., 2025 — arXiv:2502.10391 — https://arxiv.org/abs/2502.10391
  120K 人工标注对齐数据 + 多模态 RM + 改进版 MM-DPO,Phase 3/4 的进阶参考。
- **[R23] Aligning Large Multimodal Models with Factually Augmented RLHF (LLaVA-RLHF)** — Sun et al., 2023 — arXiv:2309.14525 — https://arxiv.org/abs/2309.14525
  最早把 RLHF 系统用到 VLM 的工作,同时提出了 MMHal-Bench。
- **[R24] MiniCPM-V** — Yao et al., 2024 — arXiv:2408.01800 — https://arxiv.org/abs/2408.01800
  端侧小 VLM,内含 RLAIF-V 实践细节,工程视角好参考。

### 6.6 评测 Benchmark
- **[R30] VLMEvalKit** — Duan et al., 2024 — arXiv:2407.11691 — https://arxiv.org/abs/2407.11691 — repo: https://github.com/open-compass/VLMEvalKit
  本项目 Phase 6 的评测引擎,一键 80+ benchmark。
- **[R31] POPE: Evaluating Object Hallucination in Large Vision-Language Models** — Li et al., EMNLP'23 — arXiv:2305.10355 — https://arxiv.org/abs/2305.10355
  二分类「图里有没有这个物体」,object hallucination 的标准评测。
- **[R32] CHAIR: Object Hallucination in Image Captioning** — Rohrbach et al., EMNLP'18 — arXiv:1809.02156 — https://arxiv.org/abs/1809.02156
  描述里幻觉物体的比率(CHAIRi / CHAIRs)。
- **[R33] MMHal-Bench** — 见 [R23] LLaVA-RLHF 论文 — arXiv:2309.14525
  96 个图-问对、8 类问题的开放式幻觉评测。
- **[R34] MM-Vet: Evaluating LMMs for Integrated Capabilities** — Yu et al., 2023 — arXiv:2308.02490 — https://arxiv.org/abs/2308.02490
  6 大通用能力综合评测(用 GPT-4 判分),测 alignment tax 用。
- **[R35] MMBench** — Liu et al., 2023 — arXiv:2307.06281 — https://arxiv.org/abs/2307.06281
  多维度选择题式通用能力评测。

### 6.7 量化与部署
- **[R40] GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers** — Frantar et al., ICLR'23 — arXiv:2210.17323 — https://arxiv.org/abs/2210.17323
  训练后权重量化的经典方法。
- **[R41] AWQ: Activation-aware Weight Quantization** — Lin et al., MLSys'24 — arXiv:2306.00978 — https://arxiv.org/abs/2306.00978
  按激活重要性保护关键权重,部署常用。
- **[R42] Efficient Memory Management for LLM Serving with PagedAttention (vLLM)** — Kwon et al., SOSP'23 — arXiv:2309.06180 — https://arxiv.org/abs/2309.06180
  vLLM 背后的核心技术,高吞吐推理服务。

### 6.8 训练框架与工具仓库(GitHub)
- **[G1] ms-swift** — https://github.com/modelscope/ms-swift
  本项目主框架。CPT/SFT/DPO/GRPO/PPO/RM + 量化 + vLLM/SGLang 部署 + 评测,支持 Qwen-VL/LLaVA/InternVL 等 300+ MLLM。
- **[G2] lmms-finetune** — https://github.com/zjysteven/lmms-finetune
  轻量透明的 VLM 微调代码,Phase 1 学原理用。
- **[G3] Mini-LLaVA** — https://github.com/fangyuan-ksgk/Mini-LLaVA
  极简 LLaVA 实现 + notebook,Phase 0 看架构用。
- **[G4] LLaVA(官方)** — https://github.com/haotian-liu/LLaVA
  原始 LLaVA 训练代码与配方参考。
- **[G5] RLHF-V** — https://github.com/RLHF-V/RLHF-V
  含 CHAIR/MMHal 评测代码 + 段级纠错数据。
- **[G6] RLAIF-V** — https://github.com/RLHF-V/RLAIF-V
  83K 偏好数据 + 数据生成 pipeline,Phase 2 主力。
- **[G7] MM-RLHF** — https://github.com/Kwai-YuanQi/MM-RLHF
  多模态对齐数据集 + MM-DPO + 多模态 RM。
- **[G8] VLMEvalKit** — https://github.com/open-compass/VLMEvalKit
  Phase 6 评测引擎。
- **[G9] verl** — https://github.com/volcengine/verl
  生产级 RLHF(PPO/GRPO)基础设施,Phase 5 深入用。
- **[G10] OpenRLHF** — https://github.com/OpenRLHF/OpenRLHF
  另一套易读的 RLHF 训练框架。
- **[G11] LLaMA-Factory** — https://github.com/hiyouga/LLaMA-Factory
  JD 点名的另一主流微调框架,可作 ms-swift 的对照。
- **[G12] Awesome-MLLM-Hallucination** — https://github.com/showlab/Awesome-MLLM-Hallucination
  幻觉方向论文持续汇总,跟踪最新工作用。
- **[G13] Vision-Language-Models-Overview** — https://github.com/zli12321/Vision-Language-Models-Overview
  VLM 论文与模型的前沿汇总。

---

## 7. 建议阅读顺序(从零到能动手)

1. **建立框架:** [R1] 综述 → [R2] LLaVA → [R3] LLaVA-1.5(搞懂 VLM 是什么、怎么训)
2. **微调基础:** [R10] LoRA → [R11] QLoRA(搞懂高效微调)
3. **对齐核心:** [R12] InstructGPT → [R13] DPO(搞懂 RLHF 与 DPO,这是岗位重心)
4. **对齐变体:** [R14] ORPO → [R15] KTO → [R16] SimPO(横向比较)
5. **在线 RL:** [R17] PPO → [R18] GRPO → [R19] DeepSeek-R1
6. **多模态对齐落地:** [R20] RLHF-V → [R21] RLAIF-V → [R22] MM-RLHF(直接对应你要做的事)
7. **评测与部署:** [R31] POPE / [R32] CHAIR / [R34] MM-Vet → [R40] GPTQ / [R41] AWQ / [R42] vLLM
8. **基座细节:** [R9] Qwen2-VL / [R9b] Qwen2.5-VL(动手前读)

> 提示:1–3 是地基,务必精读;4–5 可先读核心、变体略读;6 是你项目的灵魂,精读;7–8 可在做到对应 Phase 时再读。
