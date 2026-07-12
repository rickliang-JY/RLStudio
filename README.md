# RLStudio

> 从 0 到强化学习。每个算法都配同一个 grid world 例子并做了可视化，在浏览器里直接运行。

[![在线访问](https://img.shields.io/badge/在线访问-RLStudio-2b6cf0)](https://rickliang-jy.github.io/RLStudio/)
[![built with marimo](https://img.shields.io/badge/built%20with-marimo-1a73e8)](https://marimo.io)
[![notebooks](https://img.shields.io/badge/notebooks-10%20章%20%2B%2010%20例子-6b6b72)](notebooks/foundations)

**在线访问 → <https://rickliang-jy.github.io/RLStudio/>**（支持中文 / English 切换）

RLStudio 跟随赵世钰《Mathematical Foundations of Reinforcement Learning》逐章把课本里的公式
做成**能跑、能调、看得见收敛**的交互式 notebook。从基本概念到 Actor-Critic，所有算法都运行在
**同一个 grid world** 上，方便横向对比、看清差异。

## 为什么是这个项目

- **同一个网格世界** —— 十章算法共用一个 `gridworld.py`，对比不同方法在同一问题上的表现。
- **浏览器里直接跑** —— 每章一个可交互 notebook，无需本地环境，点开就能改参数、重算。
- **看得见的收敛** —— 策略、值函数、轨迹都做了可视化，拖滑块即可观察算法一步步逼近最优。
- **从数学到代码** —— 公式旁边就是能运行的实现，理论与代码一一对应。

## 内容

网站按分区组织，对应本仓库的 notebook：

| 分区 | 说明 | 状态 |
| --- | --- | --- |
| **基础 · 强化学习的数学原理** | 第 1–10 章核心算法实现 | ✅ |
| **书中例子复现** | 把书里经典插图例子做成可交互版 | ✅ |
| **实验场** | 自定义 grid world，随手改地图和参数 | ✅ |
| **进阶** | 现代策略优化：PPO / GRPO / DPO | 🚧 规划中 |
| **参考文献** | 项目引用的书、论文与代码仓库（含 BibTeX） | ✅ |
| **笔记** | 整体设计与原理的 PDF 讲解 | 🚧 规划中 |

### 基础章节对照

| 文件 | 章节 | 实现的算法 |
| --- | --- | --- |
| [`gridworld.py`](notebooks/foundations/gridworld.py) | 公共环境 | grid world 环境 + 模型 `P,R` + 采样接口 + 可视化 |
| `ch01_basic_concepts.py` | 1 基本概念 | 状态/动作/奖励/策略/轨迹/回报 的可视化 |
| `ch02_bellman_equation.py` | 2 贝尔曼方程 | 策略评估：闭式解 + 迭代解 |
| `ch03_bellman_optimality.py` | 3 贝尔曼最优方程 | 压缩迭代求解 BOE、γ 的影响 |
| `ch04_value_policy_iteration.py` | 4 值/策略迭代 | 值迭代、策略迭代、截断策略迭代 |
| `ch05_monte_carlo.py` | 5 蒙特卡洛 | MC Basic、Exploring Starts、ε-greedy |
| `ch06_stochastic_approximation.py` | 6 随机近似 | 增量均值、Robbins–Monro、SGD |
| `ch07_temporal_difference.py` | 7 时序差分 | TD(0)、Sarsa、Expected Sarsa、Q-learning |
| `ch08_value_function_approximation.py` | 8 值函数近似 | 线性 TD、DQN（经验回放 + 目标网络） |
| `ch09_policy_gradient.py` | 9 策略梯度 | REINFORCE（带基线） |
| `ch10_actor_critic.py` | 10 Actor-Critic | QAC、A2C（优势 / TD-error） |

每章另配一个 `chXX_examples.py` 复现书中的经典插图例子。

## 运行方式

### 在线（零安装）

直接打开 <https://rickliang-jy.github.io/RLStudio/>，选择分区进入：

- 大部分章节用 marimo 的 **WebAssembly** 在浏览器里直接运行；
- 用到 PyTorch 的章节（第 8 章 DQN）通过 [molab](https://molab.marimo.io) 的真实内核运行，可直接训练、Fork 修改。

### 本地

notebook 用 [marimo](https://marimo.io) 编写。装好依赖后即可编辑：

```bash
pip install -r requirements.txt              # marimo / numpy / matplotlib / wanderland
pip install torch                            # 仅第 8 章 DQN 需要，按需安装
marimo edit notebooks/foundations/ch07_temporal_difference.py
```

> **实验场的 Wanderland 3D lab**（`wanderland_gridworld_lab.py`）依赖第三方已发布包
> [`wanderland`](https://pypi.org/project/wanderland/)（`pip install wanderland` 即可，已含在
> `requirements.txt`）。该包的源码**不随本仓库分发**，本地 `repo/` 副本已被 `.gitignore` 忽略。

常用命令：

```bash
marimo edit  <file>.py     # 交互式编辑，改参数实时看结果（推荐）
marimo run   <file>.py     # 只读运行
marimo export html --no-include-code <file>.py -o out.html   # 导出静态 HTML
```

## Grid World 约定

- 5×5 网格，状态 `s = y*cols + x`（蓝色=目标，黄色=禁区）。
- 动作顺序：`0=上, 1=右, 2=下, 3=左, 4=原地`。
- 奖励：到达目标 `+1`，进入禁区 `−1`，撞墙 `−1`，普通一步 `0`，`γ=0.9`。
- `env.build_model()` 给出 `P[s,a,s'], R[s,a]`，供第 2–4 章 model-based 算法使用；
  `env.reset()/env.step()` 给出采样接口，供第 5–10 章 model-free 算法使用。

## 阅读主线

1. **2–4 章（已知模型）**：贝尔曼方程 → 最优方程 → 值/策略迭代。
2. **5–7 章（无模型，表格）**：蒙特卡洛 → 随机近似（数学桥梁）→ 时序差分。
3. **8–10 章（无模型，近似）**：值函数近似 / DQN → 策略梯度 → Actor-Critic。

每个 notebook 末尾都有“小结”承上启下，建议按章顺序阅读。

## 仓库结构

```text
RLStudio/
├── README.md
├── docs/                       GitHub Pages 站点
│   ├── index.html              双语落地页（分区导航 + Hero 动画）
│   └── notebook.html           notebook 查看器（WASM 嵌入 / molab 跳转）
└── notebooks/
    └── foundations/            marimo notebook
        ├── gridworld.py        公共 grid world 环境与可视化
        ├── chXX_*.py           第 1–10 章算法实现
        ├── chXX_examples.py    书中例子复现
        ├── custom_gridworld_lab.py       自定义实验场（matplotlib 表格视图）
        ├── wanderland_gridworld_lab.py   Wanderland 3D 实验场（Mo 走最优策略；需 wanderland）
        └── __marimo__/session/ 静态预览所需的会话快照（请勿删除）
```

> `repo/`（本地 Wanderland 源码副本）已被 `.gitignore` 忽略，不纳入版本控制 ——
> 运行 3D 实验场只需 `pip install wanderland`。

## 实现说明

- **CJK 字体** —— 在线 WASM 环境的 matplotlib 默认无中文字形，`gridworld.py` 会在浏览器内
  自动拉取并注册一个 CJK 字体，让图中的中文正常显示；本地运行则使用系统已装字体。
- **会话快照** —— `__marimo__/session/` 下的 JSON 是 molab 静态预览出图所需，已纳入版本控制，请勿删除。

## 参考文献

本项目基于：

```bibtex
@book{zhao2025RLBook,
  title={Mathematical Foundations of Reinforcement Learning},
  author={S. Zhao},
  year={2025},
  publisher={Springer Press}
}
```

## 致谢与许可

算法与示例均改编自赵世钰《Mathematical Foundations of Reinforcement Learning》，
课本内容版权归原作者所有；本仓库代码用于学习与教学目的。
