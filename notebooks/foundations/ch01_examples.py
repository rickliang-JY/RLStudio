import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 1 章 · 书中例子复现 (Worked Examples)

    复现赵世钰《强化学习的数学原理》**第 1 章**的插图例子，并做成可交互的。
    本章用一个 **3×3 网格世界**（状态 $s_1,\dots,s_9$）讲清最基础的概念：
    状态、动作、策略（确定性/随机）、奖励、轨迹与回报。

    | 例子 | 书中图 | 演示什么 |
    |---|---|---|
    | 1. 状态与动作 | Fig 1.3 | 每个状态有 5 个动作，分别去哪、得多少奖励 |
    | 2. 确定性策略 + 轨迹 | Fig 1.4 | 箭头表示策略，沿策略走出一条轨迹 |
    | 3. 随机策略 | Fig 1.5 | $s_1$ 处 0.5/0.5 向右或向下，轨迹会分叉 |
    | 4. 两个策略的回报对比 | Fig 1.6 | 绕进禁区会吃 −1，回报更低 |

    > 本章 3×3 网格设定：目标 $s_9$，禁区 $s_6,s_7$；
    > $r_{\text{boundary}}=r_{\text{forbidden}}=-1,\ r_{\text{target}}=+1,\ r_{\text{other}}=0,\ \gamma=0.9$。
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from gridworld import GridWorld, ACTION_NAMES
    return ACTION_NAMES, GridWorld, matplotlib, np, plt


@app.cell
def _(GridWorld, np):
    def ch1_grid():
        return GridWorld(
            cols=3, rows=3, start=(0, 0), target=(2, 2),
            forbidden=((2, 1), (0, 2)),
            r_target=1.0, r_forbidden=-1.0, r_step=0.0,
            r_boundary=-1.0, gamma=0.9,
        )

    def det_policy(env, mapping):
        pi = np.zeros((env.n_states, env.n_actions))
        for (x, y), a in mapping.items():
            pi[env.s_of(x, y), a] = 1.0
        return pi

    # 书中 Table 1.2 / Fig 1.4 的确定性策略 π（动作 0上 1右 2下 3左 4停）
    POLICY1 = {
        (0, 0): 1, (1, 0): 2, (2, 0): 3,
        (0, 1): 1, (1, 1): 2, (2, 1): 2,
        (0, 2): 1, (1, 2): 1, (2, 2): 4,
    }
    # Fig 1.6(b) 的“绕路”策略 π'：从 s1 往下，会穿过禁区 s7 吃一个 −1
    POLICY2 = dict(POLICY1)
    POLICY2.update({(0, 0): 2, (0, 1): 2, (0, 2): 1})
    return POLICY1, POLICY2, ch1_grid, det_policy


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · 状态与动作 (Fig 1.3)
    网格里每个格子是一个**状态** $s$；在任一状态都有 **5 个动作** $a_1..a_5$ =
    上 / 右 / 下 / 左 / 停。**选一个状态**，看这 5 个动作分别把智能体带到哪、拿到多少奖励。
    （撞墙原地不动得 $r_{\text{boundary}}=-1$；进禁区得 $-1$；到目标得 $+1$。）
    """)
    return


@app.cell
def _(ch1_grid, mo):
    _env = ch1_grid()
    s_dropdown = mo.ui.dropdown(
        options={f"s{i+1}": i for i in range(_env.n_states)},
        value="s1", label="选择状态 s")
    s_dropdown
    return (s_dropdown,)


@app.cell
def _(ACTION_NAMES, ch1_grid, mo, np, s_dropdown):
    env_sa = ch1_grid()
    _s = s_dropdown.value
    _x, _y = env_sa.xy_of(_s)
    _cn = {"up": "上", "right": "右", "down": "下", "left": "左", "stay": "停"}
    _lines = []
    for _a in range(env_sa.n_actions):
        _ns, _r = env_sa.transition(_s, _a)
        _nxy = env_sa.xy_of(_ns)
        _tag = " (撞墙原地)" if _ns == _s and _a != 4 else ""
        _lines.append(f"| a{_a+1} {_cn[ACTION_NAMES[_a]]} | s{_ns+1} {_nxy}{_tag} | {_r:+.0f} |")
    _body = "\n".join(_lines)
    fig_sa = env_sa.plot_policy(
        np.zeros((env_sa.n_states, env_sa.n_actions)),
        values=[i + 1 for i in range(env_sa.n_states)],
        title=f"3×3 网格（格内数字=状态编号 s）；当前选中 s{_s+1}={_x, _y}")
    mo.hstack([
        fig_sa,
        mo.md(f"""**在 s{_s+1} 处采取每个动作的结果**

| 动作 | 到达状态 | 奖励 r |
|---|---|---|
{_body}

目标 = s9，禁区 = s6 / s7。""")
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · 确定性策略 + 轨迹 (Fig 1.4)
    **策略** $\pi(a|s)$ 规定每个状态该做什么动作；确定性策略每格只有**一个**箭头。
    沿着箭头走就得到一条**轨迹**。**选起点**、**拖步数**，看轨迹一格格延伸到目标。
    """)
    return


@app.cell
def _(POLICY1, ch1_grid, det_policy, mo):
    _env = ch1_grid()
    start_dropdown = mo.ui.dropdown(
        options={f"s{i+1}": i for i in range(_env.n_states)},
        value="s1", label="轨迹起点")
    step_slider = mo.ui.slider(1, 8, step=1, value=8,
                               label="走多少步", show_value=True, full_width=True)
    mo.vstack([start_dropdown, step_slider])
    return start_dropdown, step_slider


@app.cell
def _(POLICY1, ch1_grid, det_policy, np, start_dropdown, step_slider):
    env_t = ch1_grid()
    pi1 = det_policy(env_t, POLICY1)

    def follow(env, pi, s0, n):
        s = s0
        traj = [s0]
        for _ in range(n):
            a = int(pi[s].argmax())
            ns, _ = env.transition(s, a)
            traj.append(ns)
            s = ns
        return traj

    _traj = follow(env_t, pi1, start_dropdown.value, step_slider.value)
    fig_t = env_t.plot_policy(pi1, title="确定性策略 π（绿色箭头）")
    _xs = np.array([env_t.xy_of(_s)[0] for _s in _traj], dtype=float)
    _ys = np.array([env_t.xy_of(_s)[1] for _s in _traj], dtype=float)
    _jit = np.linspace(-0.12, 0.12, len(_traj))
    ax_t = fig_t.axes[0]
    ax_t.plot(_xs + _jit, _ys + _jit, "-o", color="red", lw=1.5, ms=5, alpha=0.8)
    ax_t.plot(_xs[0] + _jit[0], _ys[0] + _jit[0], "*", color="blue", ms=18)
    ax_t.set_title(f"从 s{start_dropdown.value+1} 出发走 {step_slider.value} 步的轨迹")
    fig_t
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · 随机策略 (Fig 1.5)
    策略也可以是**随机**的。书中 $s_1$ 处以各 0.5 的概率向右或向下。
    每次采样得到的轨迹可能不同——**换种子**多采几条，看它们如何在 $s_1$ 处**分叉**。
    """)
    return


@app.cell
def _(mo):
    n_traj_slider = mo.ui.slider(1, 12, step=1, value=6,
                                 label="采样多少条轨迹", show_value=True,
                                 full_width=True)
    seed_slider = mo.ui.slider(0, 30, step=1, value=0,
                               label="随机种子", show_value=True, full_width=True)
    mo.vstack([n_traj_slider, seed_slider])
    return n_traj_slider, seed_slider


@app.cell
def _(POLICY1, ch1_grid, det_policy, matplotlib, n_traj_slider, np, seed_slider):
    env_s = ch1_grid()
    pi_sto = det_policy(env_s, POLICY1)
    pi_sto[env_s.s_of(0, 0)] = 0.0
    pi_sto[env_s.s_of(0, 0), 1] = 0.5      # s1 右
    pi_sto[env_s.s_of(0, 0), 2] = 0.5      # s1 下

    rng = np.random.default_rng(seed_slider.value)
    fig_s = env_s.plot_policy(pi_sto, title="随机策略 π：s1 处 0.5 右 / 0.5 下")
    ax_s = fig_s.axes[0]
    _colors = matplotlib.cm.tab10.colors
    for _k in range(n_traj_slider.value):
        _s = env_s.s_of(0, 0)
        _traj = [_s]
        for _ in range(8):
            _a = int(rng.choice(env_s.n_actions, p=pi_sto[_s]))
            _ns, _ = env_s.transition(_s, _a)
            _traj.append(_ns); _s = _ns
        _xs = np.array([env_s.xy_of(t)[0] for t in _traj], float) + rng.normal(0, 0.07, len(_traj))
        _ys = np.array([env_s.xy_of(t)[1] for t in _traj], float) + rng.normal(0, 0.07, len(_traj))
        ax_s.plot(_xs, _ys, "-o", color=_colors[_k % 10], lw=1.2, ms=4, alpha=0.7)
    ax_s.plot(0, 0, "*", color="blue", ms=18)
    fig_s
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    多条轨迹在 $s_1$ 处一半向右、一半向下，之后各自汇向目标——这就是随机策略的“分叉”。

    ## 例子 4 · 两个策略的回报对比 (Fig 1.6)
    **回报 (return)** = 沿轨迹的折扣奖励之和 $G=r_1+\gamma r_2+\gamma^2 r_3+\cdots$。
    - **策略 1**：一路 0 奖励直到目标 → 奖励序列约 $[0,0,0,1,1,\dots]$。
    - **策略 2（绕路）**：从 $s_1$ 往下，**穿过禁区 $s_7$ 吃一个 −1** → $[0,-1,0,1,1,\dots]$。
    拖动 $\gamma$，对比两者回报：绕路那条总是更低，这正是奖励在“塑造”好策略。
    """)
    return


@app.cell
def _(mo):
    gamma_slider = mo.ui.slider(0.0, 0.99, step=0.01, value=0.9,
                                label="折扣因子 γ", show_value=True, full_width=True)
    gamma_slider
    return (gamma_slider,)


@app.cell
def _(POLICY1, POLICY2, ch1_grid, det_policy, gamma_slider, mo):
    def rollout_rewards(env, pi, s0, n):
        s = s0
        traj = [s0]
        rews = []
        for _ in range(n):
            a = int(pi[s].argmax())
            ns, r = env.transition(s, a)
            traj.append(ns); rews.append(r); s = ns
        return traj, rews

    def disc(rews, g):
        G = 0.0
        for r in reversed(rews):
            G = r + g * G
        return G

    _g = gamma_slider.value
    env_a = ch1_grid()
    _figs = []
    for _name, _mapping in [("策略 1（直达）", POLICY1), ("策略 2（绕进禁区）", POLICY2)]:
        _pi = det_policy(env_a, _mapping)
        _traj, _rews = rollout_rewards(env_a, _pi, 0, 8)
        _G = disc(_rews, _g)
        _seq = ", ".join(f"{r:+.0f}" for r in _rews[:6])
        _f = env_a.plot_trajectory(
            _traj[:6], mark_current=True,
            title=f"{_name}\n奖励=[{_seq}...]  回报 G={_G:+.2f}")
        _figs.append(_f)
    mo.hstack(_figs, widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    同样的 $\gamma$ 下，策略 2 因为多吃了一个 −1，回报恒低于策略 1。
    回报就是评价一条轨迹好坏的标尺，也是后面**状态值/动作值**的基础。

    ## 小结
    - **状态 / 动作 / 策略 / 奖励 / 轨迹 / 回报**是 RL 的最基本词汇；
    - 策略可以是**确定性**（一格一箭头）或**随机**（一格多箭头，按概率走）；
    - **回报**用折扣因子 $\gamma$ 把未来奖励加权求和——它把“好策略”和“坏策略”区分开。
    下一章把“从某状态出发的期望回报”正式定义为**状态值**，并用**贝尔曼方程**求它。
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
