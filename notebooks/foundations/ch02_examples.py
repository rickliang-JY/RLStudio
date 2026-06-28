import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 2 章 · 书中例子复现 (Worked Examples)

    复现赵世钰《强化学习的数学原理》**第 2 章**的插图例子，全部做成可交互的。
    本章用一个 **2×2 网格**（$s_1,s_2,s_3,s_4$）把**回报 → 状态值 → 贝尔曼方程 → 动作值**串起来：

    | 例子 | 书中图 | 演示什么 |
    |---|---|---|
    | 1. 回报为何重要 | Fig 2.2 | 三个策略在 $s_1$ 不同，用回报比出好坏 |
    | 2. 怎么算回报（自举） | Fig 2.3 | 定义法 vs **bootstrapping** 线性方程 |
    | 3. 贝尔曼方程 | Fig 2.4 / 2.5 | 确定性 / 随机策略，解出状态值 |
    | 4. 策略 vs 状态值（迭代解） | Fig 2.7 | 在 5×5 上用迭代法 (2.11) 看 $v_k$ 收敛 |
    | 5. 从状态值到动作值 | Fig 2.8 | 算 $q_\pi(s_1,a)$，含"未选动作也有值"的常见误区 |

    > 2×2 网格设定：$s_2$ 为禁区、$s_4$ 为目标；
    > $r_{\text{target}}=1,\ r_{\text{forbidden}}=r_{\text{boundary}}=-1,\ r_{\text{other}}=0$。
    > 状态编号 $s_i$ 与坐标：$s_1=(0,0),\ s_2=(1,0),\ s_3=(0,1),\ s_4=(1,1)$。
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from gridworld import GridWorld, classic_example, greedy_policy_from_q, q_from_v

    return GridWorld, classic_example, greedy_policy_from_q, np, plt, q_from_v


@app.cell
def _(GridWorld, np):
    def grid2x2():
        # 书中第 2 章反复使用的 2×2 网格：s2 禁区、s4 目标。
        return GridWorld(
            cols=2, rows=2, start=(0, 0), target=(1, 1),
            forbidden=((1, 0),),
            r_target=1.0, r_forbidden=-1.0, r_step=0.0,
            r_boundary=-1.0, gamma=0.9,
        )

    def solve_values(env, pi, gamma):
        """闭式解 vπ = (I − γPπ)^{-1} rπ。"""
        P, R = env.build_model()
        Ppi = np.einsum("sa,sat->st", pi, P)
        rpi = np.einsum("sa,sa->s", pi, R)
        return np.linalg.solve(np.eye(env.n_states) - gamma * Ppi, rpi)

    # 动作编号：0上 1右 2下 3左 4停（= 书中 a1..a5）
    def det_policy(env, mapping):
        pi = np.zeros((env.n_states, env.n_actions))
        for (x, y), a in mapping.items():
            pi[env.s_of(x, y), a] = 1.0
        return pi

    return det_policy, grid2x2, solve_values


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · 回报为何重要 (Fig 2.2)
    三个策略只在 $s_1$ 处不同：**①向下**（避开禁区）、**②向右**（撞进禁区 $s_2$）、
    **③一半向右一半向下**。从 $s_1$ 出发的（折扣）回报分别是

    $$\text{return}_1=\frac{\gamma}{1-\gamma},\quad
      \text{return}_2=-1+\frac{\gamma}{1-\gamma},\quad
      \text{return}_3=-0.5+\frac{\gamma}{1-\gamma}.$$

    拖动 $\gamma$：永远有 $\text{return}_1>\text{return}_3>\text{return}_2$——数学和"避开禁区更好"的直觉一致。
    """)
    return


@app.cell
def _(mo):
    gamma22_slider = mo.ui.slider(0.05, 0.95, step=0.05, value=0.9,
                                  label="折扣因子 γ", show_value=True,
                                  full_width=True)
    gamma22_slider
    return (gamma22_slider,)


@app.cell
def _(gamma22_slider, mo, plt):
    _g = gamma22_slider.value
    _base = _g / (1 - _g)
    _r1, _r2, _r3 = _base, -1 + _base, -0.5 + _base
    _fig, _ax = plt.subplots(figsize=(5.2, 4))
    _names = ["策略①\n(向下,避禁区)", "策略③\n(0.5/0.5)", "策略②\n(向右,进禁区)"]
    _vals = [_r1, _r3, _r2]
    _cols = ["#4caf50", "#ff9800", "#e53935"]
    _bars = _ax.bar(_names, _vals, color=_cols)
    for _b, _v in zip(_bars, _vals):
        _ax.text(_b.get_x() + _b.get_width() / 2, _v,
                 f"{_v:.2f}", ha="center",
                 va="bottom" if _v >= 0 else "top", fontsize=10)
    _ax.axhline(0, color="gray", lw=0.8)
    _ax.set_ylabel("从 s1 出发的回报")
    _ax.set_title(f"γ={_g:.2f}：return1 > return3 > return2")
    _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""**γ = {_g:.2f} 时的回报**

    | 策略 | 公式 | 回报 |
    |---|---|---|
    | ① 向下 | γ/(1−γ) | **{_r1:.3f}** |
    | ③ 0.5/0.5 | −0.5+γ/(1−γ) | **{_r3:.3f}** |
    | ② 向右 | −1+γ/(1−γ) | **{_r2:.3f}** |

    排序恒为 ① > ③ > ②，与"避开禁区更好"的直觉吻合。""")
    ], widths=[3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · 怎么算回报：定义法 vs 自举 (Fig 2.3)
    一个**无目标无禁区**的 2×2 循环：$s_1\to s_2\to s_3\to s_4\to s_1\to\cdots$，
    每步拿到固定奖励 $r_1,r_2,r_3,r_4$。两种算回报的方式：

    - **定义法**：$v_i = r_i + \gamma r_{i+1} + \gamma^2 r_{i+2}+\cdots$（截断求和近似）。
    - **自举 (bootstrapping)**：回报互相依赖 $v_i = r_i + \gamma v_{i+1}$，写成
      $v = r + \gamma P v$ 一次线性求解。**两者结果一致**。

    调奖励和 $\gamma$，对比两列数字——自举只需解 4 元线性方程，截断却要加很多项。
    """)
    return


@app.cell
def _(mo):
    r1_s = mo.ui.slider(-2.0, 2.0, step=0.5, value=0.0, label="r₁", show_value=True)
    r2_s = mo.ui.slider(-2.0, 2.0, step=0.5, value=1.0, label="r₂", show_value=True)
    r3_s = mo.ui.slider(-2.0, 2.0, step=0.5, value=1.0, label="r₃", show_value=True)
    r4_s = mo.ui.slider(-2.0, 2.0, step=0.5, value=1.0, label="r₄", show_value=True)
    g23_s = mo.ui.slider(0.1, 0.95, step=0.05, value=0.9, label="γ", show_value=True)
    trunc_s = mo.ui.slider(2, 200, step=1, value=20,
                           label="定义法截断步数 N", show_value=True, full_width=True)
    mo.vstack([mo.hstack([r1_s, r2_s, r3_s, r4_s, g23_s]), trunc_s])
    return g23_s, r1_s, r2_s, r3_s, r4_s, trunc_s


@app.cell(hide_code=True)
def _(g23_s, mo, np, r1_s, r2_s, r3_s, r4_s, trunc_s):
    _r = np.array([r1_s.value, r2_s.value, r3_s.value, r4_s.value], float)
    _g = g23_s.value
    # 循环 s1->s2->s3->s4->s1，转移矩阵 P（行 i 指向 i+1）
    _P = np.array([[0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   [1, 0, 0, 0]], float)
    # 自举（线性解）
    _v_boot = np.linalg.solve(np.eye(4) - _g * _P, _r)
    # 定义法（截断求和）：沿循环展开 N 步
    _N = trunc_s.value
    _v_def = np.zeros(4)
    for _i in range(4):
        _acc, _disc, _idx = 0.0, 1.0, _i
        for _k in range(_N):
            _acc += _disc * _r[_idx]
            _disc *= _g
            _idx = (_idx + 1) % 4
        _v_def[_i] = _acc
    _rows = "\n".join(
        f"| v{_i+1} | {_v_def[_i]:.4f} | {_v_boot[_i]:.4f} | {abs(_v_def[_i]-_v_boot[_i]):.1e} |"
        for _i in range(4))
    mo.md(f"""**γ={_g:.2f}，奖励循环 (r₁,r₂,r₃,r₄)=({_r[0]:.1f},{_r[1]:.1f},{_r[2]:.1f},{_r[3]:.1f})**

    | 状态 | 定义法 (N={_N} 步截断) | 自举（线性解） | 差 |
    |---|---|---|---|
    {_rows}

    截断步数 N 越大，定义法越逼近自举的精确解——这正是"回报互相依赖"的 bootstrapping 思想。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · 贝尔曼方程：确定性 vs 随机策略 (Fig 2.4 / 2.5)
    在 2×2 网格（$s_2$ 禁区、$s_4$ 目标）上写出贝尔曼方程
    $v_\pi(s)=r_\pi(s)+\gamma\sum_{s'}p_\pi(s'|s)\,v_\pi(s')$ 并闭式求解。

    - **确定性策略 (Fig 2.4)**：$s_1$ 向下，避开禁区 → $v_\pi=[9,10,10,10]$（γ=0.9）。
    - **随机策略 (Fig 2.5)**：$s_1$ 以 0.5/0.5 向右或向下，会有概率进禁区 →
      $v_\pi(s_1)=-0.5+\frac{\gamma}{1-\gamma}=8.5$。

    确定性策略的每个状态值都 $\ge$ 随机策略——状态值可以用来评估策略好坏。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    policy3_dropdown = mo.ui.dropdown(
        options={"确定性策略 (Fig 2.4)": "det", "随机策略 (Fig 2.5)": "sto"},
        value="确定性策略 (Fig 2.4)", label="选择策略")
    gamma3_slider = mo.ui.slider(0.1, 0.95, step=0.05, value=0.9,
                                 label="折扣因子 γ", show_value=True, full_width=True)
    mo.vstack([policy3_dropdown, gamma3_slider])
    return gamma3_slider, policy3_dropdown


@app.cell(hide_code=True)
def _(det_policy, gamma3_slider, grid2x2, mo, policy3_dropdown, solve_values):
    env3 = grid2x2()
    env3.gamma = gamma3_slider.value
    # 公共部分：s2 向下到 s4，s3 向右到 s4，s4 自环停
    _base_map = {(1, 0): 2, (0, 1): 1, (1, 1): 4}
    pi3 = det_policy(env3, {**_base_map, (0, 0): 2})   # 确定性：s1 向下
    if policy3_dropdown.value == "sto":
        pi3[env3.s_of(0, 0)] = 0.0
        pi3[env3.s_of(0, 0), 1] = 0.5   # s1 右
        pi3[env3.s_of(0, 0), 2] = 0.5   # s1 下
    v3 = solve_values(env3, pi3, gamma3_slider.value)
    _kind = "确定性 (Fig 2.4)" if policy3_dropdown.value == "det" else "随机 (Fig 2.5)"
    fig3 = env3.plot_policy(
        pi3, v3, precision=2,
        title=f"{_kind}  γ={gamma3_slider.value:.2f}")
    _rows = "\n".join(f"| v(s{_i+1}) | **{v3[_i]:.2f}** |" for _i in range(4))
    mo.hstack([
        fig3,
        mo.md(f"""**闭式解 vπ = (I − γPπ)⁻¹ rπ**

    | 状态 | 状态值 |
    |---|---|
    {_rows}

    γ=0.9 时：确定性 → [9,10,10,10]；随机 → [8.5,10,10,10]。
    确定性策略每格都 ≥ 随机策略。""")
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 4 · 策略 vs 状态值：迭代求解 (Fig 2.7)
    回到经典 **5×5 网格**（$r_{\text{boundary}}=r_{\text{forbidden}}=-1,\ r_{\text{target}}=1,\ \gamma=0.9$）。
    不用闭式逆矩阵，而用迭代算法 (2.11)：$v_{k+1}=r_\pi+\gamma P_\pi v_k$，保证 $v_k\to v_\pi$。

    选一个策略，**拖动迭代步数滑块**看 $v_k$ 从全 0 逐步收敛：
    - **好策略**（最优策略）：状态值高、靠近目标处达 10；
    - **坏策略**（一律向上/随机）：很多状态值为负——印证"坏策略 → 低状态值"。
    """)
    return


@app.cell
def _(classic_example, greedy_policy_from_q, np, q_from_v):
    def policy_eval_snapshots(env, pi, gamma, n_iter=80):
        P, R = env.build_model()
        Ppi = np.einsum("sa,sat->st", pi, P)
        rpi = np.einsum("sa,sa->s", pi, R)
        v = np.zeros(env.n_states)
        snaps = [v.copy()]
        for _ in range(n_iter):
            v = rpi + gamma * Ppi.dot(v)
            snaps.append(v.copy())
        return snaps

    _env = classic_example()
    _P, _R = _env.build_model()
    _g = _env.gamma
    # 好策略 = 最优策略（值迭代求）
    _v = np.zeros(_env.n_states)
    for _ in range(1000):
        _v = (_R + _g * _P.dot(_v)).max(1)
    pi_good = greedy_policy_from_q(q_from_v(_v, _P, _R, _g))
    # 坏策略 = 一律向上（动作 0）
    pi_bad = np.zeros((_env.n_states, _env.n_actions)); pi_bad[:, 0] = 1.0
    # 随机策略 = 均匀
    pi_unif = np.ones((_env.n_states, _env.n_actions)) / _env.n_actions

    snaps_good = policy_eval_snapshots(_env, pi_good, _g)
    snaps_bad = policy_eval_snapshots(_env, pi_bad, _g)
    snaps_unif = policy_eval_snapshots(_env, pi_unif, _g)
    return pi_bad, pi_good, pi_unif, snaps_bad, snaps_good, snaps_unif


@app.cell
def _(mo, snaps_good):
    policy4_dropdown = mo.ui.dropdown(
        options={"好策略（最优）": "good", "坏策略（一律向上）": "bad",
                 "随机均匀策略": "unif"},
        value="好策略（最优）", label="选择策略")
    iter4_slider = mo.ui.slider(0, len(snaps_good) - 1, step=1, value=0,
                                label="迭代步数 k（往右迭代越久）", show_value=True,
                                full_width=True)
    mo.vstack([policy4_dropdown, iter4_slider])
    return iter4_slider, policy4_dropdown


@app.cell
def _(
    classic_example,
    iter4_slider,
    pi_bad,
    pi_good,
    pi_unif,
    policy4_dropdown,
    snaps_bad,
    snaps_good,
    snaps_unif,
):
    env4 = classic_example()
    _pick = policy4_dropdown.value
    _pi = {"good": pi_good, "bad": pi_bad, "unif": pi_unif}[_pick]
    _snaps = {"good": snaps_good, "bad": snaps_bad, "unif": snaps_unif}[_pick]
    _k = min(iter4_slider.value, len(_snaps) - 1)
    _label = {"good": "好策略（最优）", "bad": "坏策略（一律向上）",
              "unif": "随机均匀策略"}[_pick]
    env4.plot_policy(_pi, _snaps[_k], precision=1,
                     title=f"{_label} · 迭代法 v_k @ k={_k}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    把滑块拖到底（$k$ 大）即收敛到 $v_\pi$。好策略各状态值都为正、目标附近为 10；
    坏策略大量状态值为负——这正是 Fig 2.7 的结论：**状态值能区分好坏策略**。
    （书中 Fig 2.7 用的是另几条具体策略，数值不同，但奖励设定与结论一致。）

    ## 例子 5 · 从状态值到动作值 (Fig 2.8)
    动作值 $q_\pi(s,a)=\sum_r p(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')$。
    用例子 3 的**随机策略**，看 $s_1$ 处全部 5 个动作的动作值——
    **包括策略不选的 $a_1,a_4,a_5$**：它们照样有动作值，绝不能想当然设成 0（常见误区）。
    """)
    return


@app.cell
def _(det_policy, grid2x2, mo, plt, solve_values):
    env5 = grid2x2()
    _base_map = {(1, 0): 2, (0, 1): 1, (1, 1): 4}
    pi5 = det_policy(env5, _base_map)
    pi5[env5.s_of(0, 0)] = 0.0
    pi5[env5.s_of(0, 0), 1] = 0.5
    pi5[env5.s_of(0, 0), 2] = 0.5
    v5 = solve_values(env5, pi5, env5.gamma)
    _P, _R = env5.build_model()
    q5 = _R + env5.gamma * _P.dot(v5)        # q(s,a) for all s,a
    _qs1 = q5[env5.s_of(0, 0)]
    _names = ["a1 上", "a2 右*", "a3 下*", "a4 左", "a5 停"]
    _picked = [False, True, True, False, False]
    _cols = ["#1976d2" if p else "#b0bec5" for p in _picked]
    _fig, _ax = plt.subplots(figsize=(5.0, 3.2))
    _bars = _ax.bar(_names, _qs1, color=_cols)
    for _b, _v in zip(_bars, _qs1):
        _ax.text(_b.get_x() + _b.get_width() / 2, _v, f"{_v:.2f}",
                 ha="center", va="bottom" if _v >= 0 else "top", fontsize=9)
    _ax.axhline(0, color="gray", lw=0.8)
    _ax.set_ylabel("q(s1, a)")
    _ax.set_title("s1 处所有动作的动作值（*=随机策略实际会选的动作）")
    _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""**s₁ 处的动作值（vπ(s₂)=vπ(s₃)=10）**

    | 动作 | 计算 | q(s₁,a) |
    |---|---|---|
    | a₂ 右 | −1+γ·v(s₂) | **{_qs1[1]:.2f}** |
    | a₃ 下 | 0+γ·v(s₃) | **{_qs1[2]:.2f}** |
    | a₁ 上 | −1+γ·v(s₁) | {_qs1[0]:.2f} |
    | a₄ 左 | −1+γ·v(s₁) | {_qs1[3]:.2f} |
    | a₅ 停 | 0+γ·v(s₁) | {_qs1[4]:.2f} |

    **误区**：a₁,a₄,a₅ 虽不被策略选中，但**仍有动作值**，不能设成 0。""")
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结
    - **回报**能比出策略好坏（Fig 2.2），但要算回报得用**自举**：回报互相依赖，
      写成 $v=r+\gamma Pv$（Fig 2.3）。
    - 把"从某状态出发的期望回报"定义为**状态值**，它满足**贝尔曼方程**
      $v_\pi=r_\pi+\gamma P_\pi v_\pi$，可闭式解也可迭代解（Fig 2.4/2.5/2.7）。
    - **动作值** $q_\pi(s,a)$ 衡量在某状态选某动作的好坏；
      $v_\pi(s)=\sum_a\pi(a|s)q_\pi(s,a)$，未选动作也有值（Fig 2.8）。
    下一章用动作值定义**最优策略**与**贝尔曼最优方程**。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
