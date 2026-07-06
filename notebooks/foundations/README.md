# 强化学习数学原理 · 算法实现 (marimo)

本目录用 [marimo](https://marimo.io) 交互式 notebook，逐章实现赵世钰《强化学习的数学原理》
（*Mathematical Foundation of Reinforcement Learning*）第 1–10 章的核心算法，
**每个算法都配同一个 grid world 例子并可视化**，边讲原理边看效果。

## 文件结构

| 文件 | 章节 | 实现的算法 |
|---|---|---|
| `gridworld.py` | 公共环境 | grid world 环境 + 模型 `P,R` + 采样接口 + 可视化 |
| `ch01_basic_concepts.py` | 1 基本概念 | 状态/动作/奖励/策略/轨迹/回报 的可视化 |
| `ch02_bellman_equation.py` | 2 贝尔曼方程 | 策略评估：闭式解 + 迭代解 |
| `ch03_bellman_optimality.py` | 3 贝尔曼最优方程 | 压缩迭代求解 BOE、γ 的影响 |
| `ch04_value_policy_iteration.py` | 4 值/策略迭代 | 值迭代、策略迭代、截断策略迭代 |
| `ch05_monte_carlo.py` | 5 蒙特卡洛 | MC Basic、Exploring Starts、ε-greedy |
| `ch06_stochastic_approximation.py` | 6 随机近似 | 增量均值、Robbins–Monro、SGD |
| `ch07_temporal_difference.py` | 7 时序差分 | TD(0)、Sarsa、Expected Sarsa、Q-learning |
| `ch08_value_function_approximation.py` | 8 值函数近似 | 线性 TD、DQN（经验回放 + 目标网络）|
| `ch09_policy_gradient.py` | 9 策略梯度 | REINFORCE（带基线）|
| `ch10_actor_critic.py` | 10 Actor-Critic | QAC、A2C（优势/TD-error）|

## 运行方式

本项目使用专用的 conda 环境 `marimo`（含 marimo / numpy / matplotlib / torch）。

```bash
# 进入本目录
cd marimo_rl_algorithms

# 交互式编辑（推荐，可改参数实时看结果）
D:\Program\Anaconda\envs\marimo\python.exe -m marimo edit ch07_temporal_difference.py

# 或只读运行
D:\Program\Anaconda\envs\marimo\python.exe -m marimo run ch07_temporal_difference.py

# 导出为静态 HTML
D:\Program\Anaconda\envs\marimo\python.exe -m marimo export html --no-include-code ch07_temporal_difference.py -o ch07.html
```

> 也可以 `conda activate marimo` 后直接 `marimo edit <file>.py`。

## Grid World 约定

- 5×5 网格，状态 `s = y*cols + x`（蓝色=目标，黄色=禁区）。
- 动作顺序：`0=上, 1=右, 2=下, 3=左, 4=原地`。
- 奖励：到达目标 `+1`，进入禁区 `−1`，撞墙 `−1`，普通一步 `0`，`γ=0.9`。
- `env.build_model()` 给出 `P[s,a,s'], R[s,a]`，供第 2–4 章 model-based 算法使用；
  `env.reset()/env.step()` 给出采样接口，供第 5–10 章 model-free 算法使用。

## 阅读主线

1. **2–4 章（已知模型）**：贝尔曼方程 → 最优方程 → 值/策略迭代。
2. **5–7 章（无模型，表格）**：蒙特卡洛 → 随机近似（数学桥梁）→ 时序差分。
3. **8–10 章（无模型，近似）**：值函数近似/DQN → 策略梯度 → Actor-Critic。

每个 notebook 末尾都有"小结"承上启下，建议按章顺序阅读。
