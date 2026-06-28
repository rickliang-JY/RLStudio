# RLStudio

从 0 到强化学习。每个算法都配同一个 grid world 例子并可视化，在浏览器里直接运行。

**在线访问 → <https://rickliang-jy.github.io/RLStudio/>**

## 内容

- **基础 · 强化学习的数学原理** —— 赵世钰《Mathematical Foundation of Reinforcement
  Learning》第 1–10 章算法实现（贝尔曼方程、值/策略迭代、蒙特卡洛、时序差分、
  值函数近似/DQN、策略梯度、Actor-Critic）。
- **书中例子复现** —— 把书里经典插图例子做成可交互版。
- **实验场** —— 自定义 grid world，随手改地图和参数。
- **进阶（规划中）** —— PPO / GRPO / DPO。
- **笔记** —— PDF 设计与原理讲解。

## 仓库结构

```text
docs/                  GitHub Pages 落地页（index.html + notebook.html）
notebooks/
  foundations/         marimo notebook（gridworld.py + 各章 + 书中例子 + lab）
    __marimo__/session/  静态预览所需的会话快照（请勿删除）
```

## 本地运行 notebook

notebook 用 [marimo](https://marimo.io) 编写。装好 marimo 后：

```bash
marimo edit notebooks/foundations/ch07_temporal_difference.py
```

在线运行无需安装：网站每章用 marimo 的 WebAssembly 渲染直接在浏览器里跑；
用到 PyTorch 的章节（第 8 章 DQN）通过 [molab](https://molab.marimo.io) 的真实内核运行。
