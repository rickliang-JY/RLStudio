# VLM 后训练全流程 · 实战子项目

> RLStudio 的进阶子项目。foundations 把经典 RL 算法做成「看得见的收敛」;
> 本子项目把这条线延伸到**现代对齐实战**——拿一个小 VLM,把后训练全生命周期
> (SFT → 偏好对齐 → 奖励模型 → 在线 RL → 评测 → 量化部署)在一张 RTX 5090 上从头跑通。

完整技术地图与论文清单见 [`docs/VLM后训练_技术档案.md`](../docs/VLM后训练_技术档案.md)。

## 定位

- **目标**:不追求创新,追求**全面与扎实**——把后训练每一站都站在成熟开源仓库上跑通跑对。
- **基座**:`Qwen2-VL-2B-Instruct`(主线,单卡全流程不挤);打通后可上 `Qwen2.5-VL-7B`(LoRA)做加强版。
- **主框架**:`ms-swift`(一个框架覆盖 SFT/DPO/GRPO/PPO/RM + 量化 + 部署 + 评测)。
- **硬件**:单张 RTX 5090(32GB,Blackwell / sm_120)。环境配置见 [`env/README.md`](env/README.md)。

## 阶段索引

| Phase | 主题 | 脚本 | 状态 |
|---|---|---|---|
| [0](phase0_architecture/) | 看懂结构:ViT + connector + LLM | `env/setup_autodl.sh` | 🚧 脚本就绪,待 AutoDL 验证 |
| [1](phase1_sft/) | SFT 视觉指令微调(LoRA) | `phase1_sft/run_sft.sh` | 🚧 脚本就绪 |
| [2](phase2_preference_data/) | 偏好数据(RLAIF-V) | `phase2_preference_data/run_prepare_data.sh` | 🚧 脚本就绪 |
| [3](phase3_dpo/) | DPO / ORPO / KTO ★JD 核心 | `phase3_dpo/run_align.sh {dpo,orpo,kto}` | 🚧 脚本就绪 |
| [4](phase4_reward_model/) | 奖励模型 | `phase4_reward_model/run_rm.sh` | 🚧 脚本就绪 |
| [5](phase5_online_rl/) | 在线 RL:GRPO | `phase5_online_rl/run_grpo.sh` | 🚧 脚本就绪 |
| [6](phase6_eval/) | 评测(POPE/MMBench/MM-Vet) | `phase6_eval/run_eval.sh` | 🚧 脚本就绪 |
| [7](phase7_quant_deploy/) | 量化(AWQ/GPTQ)+ vLLM 部署 | `phase7_quant_deploy/run_quant_deploy.sh` | 🚧 脚本就绪 |

> 8 个阶段是**渐进**的,每做完一个就多一项能写进简历的东西。即便只到 Phase 3,也已覆盖 JD 核心。
> 脚本均按 ms-swift 文档化用法、针对单卡 5090 + Qwen2-VL-2B 写好默认参数;Phase 0 验证、确认 ms-swift 版本后,
> 个别参数(数据集 id、显存档位)可能需微调。

## 运行顺序(在 AutoDL 上)

```bash
# 0) 拉代码(数据盘)
cd /root/autodl-tmp && git clone https://github.com/rickliang-JY/RLStudio.git || git -C RLStudio pull
cd RLStudio/vlm_posttraining

# 0) Phase 0:环境验证 + 推理 sanity(不装训练框架)
bash env/setup_autodl.sh

# 1) 装框架(SFT/DPO/RM 够用;Phase 5/6/7 再 `bash env/setup_swift.sh full`)
bash env/setup_swift.sh

# 2) 全流程
bash phase1_sft/run_sft.sh                         # SFT → merge 出 qwen2vl2b-sft
bash phase2_preference_data/run_prepare_data.sh    # 看偏好数据
for t in dpo orpo kto; do bash phase3_dpo/run_align.sh $t; done
bash phase4_reward_model/run_rm.sh
bash env/setup_swift.sh full                        # 装 vllm + evalscope
bash phase5_online_rl/run_grpo.sh
bash phase6_eval/run_eval.sh                        # 横扫各 checkpoint
bash phase7_quant_deploy/run_quant_deploy.sh quant  # 量化;再 deploy / bench
```

> 各阶段产物默认落在数据盘 `/root/autodl-tmp/vlm/`(adapter / merge 模型 / 日志 / 评测结果),路径与超参见 `env/common.sh`。

## 交付物

- 本目录:每个 Phase 一个目录 + README + 可复现脚本/配置。
- 一份技术报告:把 base / SFT / DPO / GRPO 各阶段的 benchmark 数字列表对比,讲清每一步的变化。

## 单卡(5090)的范围说明

- 显存 32GB 跑 2B 全流程绰绰有余;7B 走 LoRA/QLoRA 可行,7B 的 PPO/GRPO 偏紧但配合 vLLM colocate 可跑。
- 档案里「DeepSpeed ZeRO 多卡 + SLURM 多节点」在单卡无法真正实操;本项目用 **ZeRO-2 / offload(单卡)** + **vLLM 加速**替代,分布式按「熟悉原理」如实呈现。

## 工程约定(贯穿全程)

- 实验记录用 **W&B**;每个 Phase 的 checkpoint 命名 `{base}-{phase}-{method}`,供 Phase 6 统一评测。
- 重产物(模型权重、数据、checkpoint、wandb 日志)一律 gitignore,仓库只留脚本/配置/报告。
