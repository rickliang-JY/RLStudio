import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 3 章 · 书中例子复现 (Worked Examples)

    复现赵世钰《强化学习的数学原理》**第 3 章**的插图例子，全部做成可交互的。
    本章主题：**最优状态值**与**贝尔曼最优方程 (BOE)**。

    | 例子 | 书中图 | 演示什么 |
    |---|---|---|
    | 1. 策略改进 | Fig 3.2 | 用动作值把"向右进禁区"改成"向下避禁区" |
    | 2. 影响最优策略的因素 | Fig 3.4 | 拖 γ 与 r_forbidden，最优策略/状态值随之变（远视↔短视） |
    | 3. 最优策略不唯一 | Fig 3.3 | $v^*$ 唯一，但并列动作让最优策略可有多个 |
    | 4. 不绕远路 | Fig 3.5 | 即使绕路没有负奖励，折扣因子也会惩罚绕路 |
    """)
    return


@app.cell
def _():
    import numpy as np
    from gridworld import GridWorld, classic_example, greedy_policy_from_q, q_from_v
    return (
        GridWorld,
        classic_example,
        greedy_policy_from_q,
        np,
        q_from_v,
    )


@app.cell
def _(np):
    def value_iteration(env, gamma, tol=1e-12, max_iter=5000):
        P, R = env.build_model()
        v = np.zeros(env.n_states)
        for _ in range(max_iter):
            vn = (R + gamma * P.dot(v)).max(1)
            if np.max(np.abs(vn - v)) < tol:
                break
            v = vn
        return v

    def det_policy(env, mapping):
        pi = np.zeros((env.n_states, env.n_actions))
        for (x, y), a in mapping.items():
            pi[env.s_of(x, y), a] = 1.0
        return pi

    def solve_values(env, pi, gamma):
        P, R = env.build_model()
        Ppi = np.einsum("sa,sat->st", pi, P)
        rpi = np.einsum("sa,sa->s", pi, R)
        return np.linalg.solve(np.eye(env.n_states) - gamma * Ppi, rpi)

    return det_policy, solve_values, value_iteration


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · 策略改进 (Fig 3.2)
    2×2 网格（$s_2$ 禁区、$s_4$ 目标）。给定的初始策略在 $s_1$ 选 **a₂(向右)**——
    会撞进禁区，不好。怎么改进？算出动作值 $q_\pi(s_1,a)$，**贪心地选最大的那个**。

    给定策略下（γ=0.9）：$v_\pi=[8,10,10,10]$，于是
    $q_\pi(s_1,a_3{\rm 下})=0+\gamma\cdot10=9 > q_\pi(s_1,a_2{\rm 右})=-1+\gamma\cdot10=8$。
    所以把 $s_1$ 改成**向下**——这一步就是 value/policy iteration 的核心。
    """)
    return


@app.cell
def _(mo):
    improve_switch = mo.ui.switch(value=False, label="执行策略改进（s₁ 改向下）")
    improve_switch
    return (improve_switch,)


@app.cell
def _(GridWorld, det_policy, improve_switch, mo, q_from_v, solve_values):
    env1 = GridWorld(cols=2, rows=2, start=(0, 0), target=(1, 1),
                     forbidden=((1, 0),), r_target=1.0, r_forbidden=-1.0,
                     r_step=0.0, r_boundary=-1.0, gamma=0.9)
    _base = {(1, 0): 2, (0, 1): 1, (1, 1): 4}   # s2 下, s3 右, s4 停
    _s1_action = 2 if improve_switch.value else 1   # 改进后向下，否则向右
    pi1 = det_policy(env1, {**_base, (0, 0): _s1_action})
    v1 = solve_values(env1, pi1, env1.gamma)
    _P, _R = env1.build_model()
    q1 = q_from_v(v1, _P, _R, env1.gamma)
    _qs1 = q1[env1.s_of(0, 0)]
    _kind = "改进后 π′（s1 向下）" if improve_switch.value else "初始 π（s1 向右，进禁区）"
    fig1 = env1.plot_policy(pi1, v1, precision=1, title=_kind)
    mo.hstack([
        fig1,
        mo.md(f"""**s1 处各动作的动作值 q(s1, a)**

| 动作 | q(s1,a) |
|---|---|
| a2 右（进禁区） | {_qs1[1]:.1f} |
| a3 下（避禁区） | **{_qs1[2]:.1f}** ← 最大 |
| a1 上 | {_qs1[0]:.1f} |
| a4 左 | {_qs1[3]:.1f} |
| a5 停 | {_qs1[4]:.1f} |

贪心改进选 **a3 下**。打开上面的开关看改进后的策略。
当前 v(s1) = **{v1[env1.s_of(0,0)]:.1f}**。""")
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · 影响最优策略的因素 (Fig 3.4) — 🎛️ 拖 γ 与 r_forbidden
    经典 5×5 网格。书中通过改 **折扣率 γ** 和 **禁区惩罚 r_forbidden** 演示最优策略如何变化。
    本格用值迭代实时求最优策略与最优状态值，**精确复现书中四张子图**：

    - **γ=0.90, r_f=−1**（图 a，基准）：γ 大→**远视**，敢穿禁区抄近路。
    - **γ=0.50, r_f=−1**（图 b）：γ 变小→**短视**，宁可绕远也躲禁区。
    - **γ=0.00, r_f=−1**（图 c）：完全短视，只看眼前奖励，到不了目标。
    - **γ=0.90, r_f=−10**（图 d）：重罚禁区→**彻底绕开**所有禁区。
    """)
    return


@app.cell
def _(mo):
    gamma2_slider = mo.ui.slider(0.0, 0.95, step=0.05, value=0.9,
                                 label="折扣率 γ", show_value=True, full_width=True)
    rforb_slider = mo.ui.slider(-10.0, 0.0, step=1.0, value=-1.0,
                                label="禁区惩罚 r_forbidden", show_value=True,
                                full_width=True)
    mo.vstack([gamma2_slider, rforb_slider])
    return gamma2_slider, rforb_slider


@app.cell
def _(
    GridWorld,
    gamma2_slider,
    greedy_policy_from_q,
    q_from_v,
    rforb_slider,
    value_iteration,
):
    env2 = GridWorld(
        cols=5, rows=5, start=(0, 0), target=(2, 3),
        forbidden=((1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)),
        r_target=1.0, r_forbidden=rforb_slider.value, r_step=0.0,
        r_boundary=-1.0, gamma=gamma2_slider.value)
    _g = max(gamma2_slider.value, 1e-9)   # γ=0 时用极小值避免奇异，效果等价
    v2 = value_iteration(env2, _g)
    _P, _R = env2.build_model()
    pi2 = greedy_policy_from_q(q_from_v(v2, _P, _R, _g))
    _tag = ""
    if abs(gamma2_slider.value - 0.9) < 1e-9 and rforb_slider.value == -1:
        _tag = "（= 书图 a 基准）"
    elif abs(gamma2_slider.value - 0.5) < 1e-9 and rforb_slider.value == -1:
        _tag = "（= 书图 b）"
    elif gamma2_slider.value == 0.0 and rforb_slider.value == -1:
        _tag = "（= 书图 c）"
    elif abs(gamma2_slider.value - 0.9) < 1e-9 and rforb_slider.value == -10:
        _tag = "（= 书图 d）"
    env2.plot_policy(pi2, v2, precision=1,
                     title=f"最优策略 π*  γ={gamma2_slider.value:.2f}, "
                           f"r_f={rforb_slider.value:.0f} {_tag}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **最优状态值的空间规律**：离目标越近值越大、越远越小——折扣率使长轨迹的回报变小。
    试着把滑块调到上面四组数值，会和书中 Fig 3.4(a)-(d) 的数字逐格吻合。

    ## 例子 3 · 最优策略不唯一 (Fig 3.3)
    重要结论：**最优状态值 $v^*$ 唯一，但最优策略不一定唯一**。
    当某状态有**两个动作的 $q^*(s,a)$ 并列最大**时，任选其一都给出最优策略。
    下面在经典 5×5 网格里找出所有"并列"状态，构造**两个不同但都最优**的策略，
    它们的状态值完全相同。
    """)
    return


@app.cell
def _(classic_example, np, q_from_v, value_iteration):
    env3 = classic_example()
    _g = env3.gamma
    v3 = value_iteration(env3, _g)
    _P, _R = env3.build_model()
    q3 = q_from_v(v3, _P, _R, _g)

    def greedy_pick(q, tie_break):
        """tie_break=0 取第一个最大动作；=1 取最后一个最大动作。"""
        nS, nA = q.shape
        pi = np.zeros((nS, nA))
        for s in range(nS):
            mx = q[s].max()
            best = [a for a in range(nA) if abs(q[s, a] - mx) < 1e-6]
            pi[s, best[0] if tie_break == 0 else best[-1]] = 1.0
        return pi

    pi3_a = greedy_pick(q3, 0)
    pi3_b = greedy_pick(q3, 1)
    n_diff = int((pi3_a.argmax(1) != pi3_b.argmax(1)).sum())
    return env3, n_diff, pi3_a, pi3_b, v3


@app.cell
def _(env3, mo, n_diff, pi3_a, pi3_b, solve_values, v3):
    _g = env3.gamma
    _va = solve_values(env3, pi3_a, _g)
    _vb = solve_values(env3, pi3_b, _g)
    _fa = env3.plot_policy(pi3_a, _va, precision=1, title="最优策略 A（并列时取动作①）")
    _fb = env3.plot_policy(pi3_b, _vb, precision=1, title="最优策略 B（并列时取另一个）")
    mo.vstack([
        mo.md(f"两个策略在 **{n_diff}** 个状态上动作不同，但状态值最大差异 = "
              f"**{abs(_va-_vb).max():.2e}**（即完全相同）。"),
        mo.hstack([_fa, _fb], widths=[1, 1])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    两张图箭头不同（差异处），但每格的状态值一字不差——印证"$v^*$ 唯一，$\pi^*$ 可不唯一"。

    ## 例子 4 · 不绕远路 (Fig 3.5) — 🎛️ 拖 γ
    2×2 网格、**无禁区**、目标在右下 $s_4$。比较两个只在 $s_2$ 不同的策略：
    - **直达**：$s_2\to s_4$，回报 $=1+\gamma+\gamma^2+\cdots=\frac{1}{1-\gamma}$。
    - **绕路**：$s_2\to s_1\to s_3\to s_4$，回报 $=\gamma^2\cdot\frac{1}{1-\gamma}$。

    即使绕路途中**没有任何负奖励**，折扣因子也让绕路回报更小。
    所以"为了让智能体尽快到达就得加每步 −1 惩罚"是个误解——折扣率已经在起作用。
    """)
    return


@app.cell
def _(mo):
    gamma4_slider = mo.ui.slider(0.1, 0.95, step=0.05, value=0.9,
                                 label="折扣率 γ", show_value=True, full_width=True)
    gamma4_slider
    return (gamma4_slider,)


@app.cell
def _(GridWorld, det_policy, gamma4_slider, mo, solve_values):
    env4 = GridWorld(cols=2, rows=2, start=(0, 0), target=(1, 1),
                     forbidden=(), r_target=1.0, r_forbidden=-1.0,
                     r_step=0.0, r_boundary=-1.0, gamma=gamma4_slider.value)
    _g = gamma4_slider.value
    # 直达：s2 向下；绕路：s2 向左
    pi_direct = det_policy(env4, {(0, 0): 2, (1, 0): 2, (0, 1): 1, (1, 1): 4})
    pi_detour = det_policy(env4, {(0, 0): 2, (1, 0): 3, (0, 1): 1, (1, 1): 4})
    v_d = solve_values(env4, pi_direct, _g)
    v_t = solve_values(env4, pi_detour, _g)
    _fd = env4.plot_policy(pi_direct, v_d, precision=2,
                           title=f"直达 (s2→下)  v(s2)={v_d[env4.s_of(1,0)]:.2f}")
    _ft = env4.plot_policy(pi_detour, v_t, precision=2,
                           title=f"绕路 (s2→左)  v(s2)={v_t[env4.s_of(1,0)]:.2f}")
    _direct = 1 / (1 - _g)
    _det = _g**2 / (1 - _g)
    mo.vstack([
        mo.md(f"γ={_g:.2f}：直达回报 1/(1−γ) = **{_direct:.2f}**，"
              f"绕路回报 γ²/(1−γ) = **{_det:.2f}**。绕路恒小于直达。"),
        mo.hstack([_fd, _ft], widths=[1, 1])
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结
    - **策略改进**：用动作值贪心地替换动作，状态值单调不减（Fig 3.2）。
    - **最优策略由 r 和 γ 决定**：γ 大→远视敢抄近路，γ 小→短视绕开风险；
      重罚禁区可彻底避开；奖励的仿射变换不改变最优策略（Fig 3.4）。
    - $v^*$ **唯一**但 $\pi^*$ **可不唯一**（并列动作，Fig 3.3）。
    - 折扣率天然惩罚绕路，无需额外的每步负奖励（Fig 3.5）。
    下一章把"策略改进 + 策略评估"组合成 **值迭代 / 策略迭代** 算法。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
