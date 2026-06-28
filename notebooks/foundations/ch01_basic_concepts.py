import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 1 章 · 基本概念 (Basic Concepts)

    本章不介绍"算法",而是把强化学习的**语言**讲清楚，并用 grid world 把每个
    概念可视化。后面九章的所有算法都建立在这些概念之上。

    ## Grid World 是什么
    - **状态 (state)** $s$：智能体所在的格子。$5\times5$ 共 25 个状态，
      编号 $s = y\cdot \text{cols} + x$。
    - **动作 (action)** $a$：`上、右、下、左、原地不动` 共 5 个。
    - **奖励 (reward)** $r$：到达目标 $+1$，进入禁区 $-1$，撞墙 $-1$，其余 $-1$。
    - **策略 (policy)** $\pi(a\mid s)$：在每个状态选择各动作的概率。
    - **回报 (return)** $G_t=\sum_{k\ge0}\gamma^k r_{t+k+1}$：一条轨迹未来折扣奖励之和。

    蓝色格子是**目标**，黄色格子是**禁区**。
    """)
    return


@app.cell
def _():
    import numpy as np
    from gridworld import classic_example, ACTIONS, ACTION_NAMES
    env = classic_example()
    return ACTIONS, ACTION_NAMES, env, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1. 环境布局：状态 / 目标 / 禁区
    """)
    return


@app.cell
def _(env):
    fig0 = env.plot(title="Grid World 5x5")[0]
    fig0
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. 策略 (policy)
    策略用一个 $25\times5$ 的矩阵表示，每一行是该状态下 5 个动作的概率。
    下面画一个**均匀随机策略**：每个动作 $1/5$ 概率（绿色箭头长度 $\propto$ 概率）。
    """)
    return


@app.cell
def _(ACTIONS, env, np):
    def random_policy(env):
        return np.ones((env.n_states, env.n_actions)) / env.n_actions

    def toward_target_policy(env):
        pi = np.zeros((env.n_states, env.n_actions))
        tx, ty = env.target
        for s in range(env.n_states):
            x, y = env.xy_of(s)
            best_a, best_d = 4, abs(x - tx) + abs(y - ty)
            for a, (dx, dy) in enumerate(ACTIONS):
                nx, ny = x + dx, y + dy
                if env.in_bounds(nx, ny) and abs(nx - tx) + abs(ny - ty) < best_d:
                    best_d, best_a = abs(nx - tx) + abs(ny - ty), a
            pi[s, best_a] = 1.0
        return pi

    POLICIES = {"均匀随机策略": random_policy(env),
                "朝目标走的策略": toward_target_policy(env)}
    return POLICIES, random_policy


@app.cell
def _(env, random_policy):
    fig1 = env.plot_policy(random_policy(env), title="均匀随机策略")
    fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### 3. 🎛️ 交互：选策略、调 γ、**逐步回放**一条轨迹

        给定策略，从起点不断 `step` 得到一条轨迹
        $s_0,a_0,r_1,s_1,a_1,r_2,\dots$。下面：

        - **选策略 / 改随机种子**：换一条不同的轨迹；
        - **调 γ**：实时重算这条轨迹的折扣回报 $G_0$；
        - **拖动「走到第几步」**：红圈标出智能体**此刻**的位置，
          右侧文字同步显示到目前为止累计的折扣回报。
        """
    )
    return


@app.cell
def _(POLICIES, mo):
    traj_policy = mo.ui.dropdown(options=list(POLICIES.keys()),
                                 value="均匀随机策略", label="用哪个策略采样轨迹")
    gamma_slider = mo.ui.slider(0.1, 0.99, step=0.01, value=0.9,
                                label="折扣因子 γ", show_value=True, full_width=True)
    seed_slider = mo.ui.slider(0, 30, step=1, value=1,
                               label="随机种子（换一条轨迹）", show_value=True,
                               full_width=True)
    mo.vstack([traj_policy, gamma_slider, seed_slider])
    return gamma_slider, seed_slider, traj_policy


@app.cell
def _(POLICIES, env, np, seed_slider, traj_policy):
    def rollout(env, policy, start=None, steps=30, seed=None):
        rng = np.random.default_rng(seed)
        s = env.reset(start)
        traj, actions, rewards = [s], [], []
        for _ in range(steps):
            a = rng.choice(env.n_actions, p=policy[s])
            s, r, done, _ = env.step(a)
            traj.append(s)
            actions.append(a)
            rewards.append(r)
            if done:
                break
        return traj, actions, rewards

    cur_policy = POLICIES[traj_policy.value]
    traj, actions, rewards = rollout(env, cur_policy, start=(0, 0),
                                     steps=25, seed=seed_slider.value)
    n_steps = len(rewards)
    return n_steps, rewards, traj


@app.cell
def _(mo, n_steps):
    step_slider = mo.ui.slider(0, n_steps, step=1, value=n_steps,
                               label="走到第几步 k", show_value=True, full_width=True)
    step_slider
    return (step_slider,)


@app.cell
def _(env, gamma_slider, mo, rewards, step_slider, traj):
    k = min(step_slider.value, len(traj) - 1)
    partial = traj[:k + 1]
    g = gamma_slider.value
    G_so_far = sum((g ** i) * r for i, r in enumerate(rewards[:k]))
    G_full = sum((g ** i) * r for i, r in enumerate(rewards))
    _fig = env.plot_trajectory(
        partial, mark_current=True,
        title=f"轨迹回放 第 {k}/{len(traj) - 1} 步 (γ={g:.2f})")
    mo.hstack([
        _fig,
        mo.md(
            f"""
            **当前步** k = {k}

            **到此累计回报**
            $G_0^{{(0:{k})}} = {G_so_far:.3f}$

            **整条轨迹回报**
            $G_0 = {G_full:.3f}$

            （红圈 = 智能体此刻位置；
            蓝星 = 起点）
            """
        ),
    ], justify="start", align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4. 概念小结
    | 符号 | 含义 | 代码 |
    |---|---|---|
    | $s$ | 状态 | `env.s_of(x,y)` / `env.xy_of(s)` |
    | $a$ | 动作 | `0..4` = 上右下左停 |
    | $p(s'\mid s,a)$ | 状态转移 | `env.build_model()` 返回 `P` |
    | $r(s,a)$ | 期望奖励 | `env.build_model()` 返回 `R` |
    | $\pi(a\mid s)$ | 策略 | `policy[s, a]` |
    | $G_t$ | 回报 | $\sum_k \gamma^k r_{t+k+1}$ |

    模型 `P, R` 是**第 2-4 章**(model-based)的输入；而 `env.step()` 采样接口是
    **第 5-10 章**(model-free)的输入。下一章用 `P, R` 求解贝尔曼方程，
    计算一个策略到底"有多好"。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
