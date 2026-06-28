import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 4 章 · 书中例子复现 (Worked Examples)

    复现赵世钰《强化学习的数学原理》**第 4 章**的插图例子，全部做成可交互的。
    本章是**求解贝尔曼最优方程**的三种算法：值迭代、策略迭代、截断策略迭代。

    | 例子 | 书中图 | 演示什么 |
    |---|---|---|
    | 1. 值迭代逐步执行 | Fig 4.2 / Table 4.1-4.3 | 每步算 q-表 → 贪心更新策略与状态值 |
    | 2. 策略迭代（简单两状态） | Fig 4.3 | 策略评估迭代收敛到 (−10,−9)，一步改进即最优 |
    | 3. 策略迭代演化（5×5） | Fig 4.4 | 离目标近的状态先收敛，逐步铺到远处 |
    | 4. 三种算法对比 | Fig 4.5 | 截断 PI 介于值迭代 (j=1) 与策略迭代 (j=∞) 之间 |
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from gridworld import GridWorld, classic_example, greedy_policy_from_q, q_from_v
    return (
        GridWorld,
        classic_example,
        greedy_policy_from_q,
        np,
        plt,
        q_from_v,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · 值迭代逐步执行 (Fig 4.2 / Tables 4.1–4.3)
    2×2 网格（$s_2$ 禁区、$s_4$ 目标，γ=0.9）。值迭代每一步：对所有 $(s,a)$ 算
    $q_k(s,a)=r+\gamma\sum_{s'}p(s'|s,a)v_k(s')$，再令 $v_{k+1}(s)=\max_a q_k(s,a)$、
    $\pi_{k+1}(s)=\arg\max_a q_k(s,a)$。**拖动 $k$** 看 q-表、策略、状态值同步更新。

    书中：$v_0=0 \Rightarrow v_1=[0,1,1,1]\Rightarrow v_2=[0.9,1.9,1.9,1.9]$，$\pi_2$ 已最优。
    """)
    return


@app.cell
def _(GridWorld, np):
    def vi_snapshots(env, gamma, n_steps=8):
        P, R = env.build_model()
        v = np.zeros(env.n_states)
        snaps = []
        for _ in range(n_steps + 1):
            q = R + gamma * P.dot(v)
            pi = np.zeros_like(q)
            pi[np.arange(env.n_states), q.argmax(1)] = 1.0
            snaps.append((v.copy(), q.copy(), pi.copy()))
            v = q.max(1)
        return snaps

    env_vi = GridWorld(cols=2, rows=2, start=(0, 0), target=(1, 1),
                       forbidden=((1, 0),), r_target=1.0, r_forbidden=-1.0,
                       r_step=0.0, r_boundary=-1.0, gamma=0.9)
    vi_snaps = vi_snapshots(env_vi, env_vi.gamma, n_steps=8)
    return env_vi, vi_snaps


@app.cell
def _(mo, vi_snaps):
    k_slider = mo.ui.slider(0, len(vi_snaps) - 1, step=1, value=0,
                            label="值迭代步数 k", show_value=True, full_width=True)
    k_slider
    return (k_slider,)


@app.cell
def _(env_vi, k_slider, mo, vi_snaps):
    _k = k_slider.value
    _vk, _qk, _pik1 = vi_snaps[_k]
    _anames = ["a1", "a2", "a3", "a4", "a5"]
    _rows = []
    for _si in range(4):
        _cells = " | ".join(f"{_qk[_si, _a]:.2f}" for _a in range(5))
        _best = _qk[_si].argmax()
        _rows.append(f"| s{_si+1} | {_cells} |  → **a{_best+1}** |")
    _qtable = "\n".join(_rows)
    _fig = env_vi.plot_policy(
        _pik1, _vk, precision=2,
        title=f"v_{_k}（格内）与据此贪心得到的 π_{_k+1}（箭头）")
    mo.hstack([
        _fig,
        mo.md(f"""**k={_k} 的 q-表 q_{_k}(s,a)**（→ 为贪心动作）

| s\\a | a1上 | a2右 | a3下 | a4左 | a5停 | argmax |
|---|---|---|---|---|---|---|
{_qtable}

状态值 v_{_k} = [{_vk[0]:.2f}, {_vk[1]:.2f}, {_vk[2]:.2f}, {_vk[3]:.2f}]

k=0→1 给出 v₁=[0,1,1,1]；k=1→2 给出 v₂≈[0.9,1.9,1.9,1.9]，π₂ 已最优。""")
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · 策略迭代：简单两状态例子 (Fig 4.3)
    一维世界，两个状态 $s_1,s_2$，三个动作 $\{a_\ell\text{左}, a_0\text{停}, a_r\text{右}\}$，
    $s_2$ 是目标。$r_{\text{boundary}}=-1,\ r_{\text{target}}=1,\ \gamma=0.9$。
    初始策略 $\pi_0$：$s_1,s_2$ 都向左——不朝目标走，很差。

    **策略评估**要解贝尔曼方程，实践中用迭代法 $v^{(j+1)}=r_\pi+\gamma P_\pi v^{(j)}$。
    拖动 $j$ 看它从 0 收敛到真值 $v_{\pi_0}=(-10,-9)$。然后**一步策略改进**就得到最优策略。
    """)
    return


@app.cell
def _(np):
    # 两状态一维 MDP：动作 0=左, 1=停, 2=右；s2(索引1) 为目标
    # s1: 左→撞墙(留s1,r=-1), 停→s1(r=0), 右→s2(r=+1)
    # s2: 左→s1(r=0), 停→s2(r=+1,目标自环), 右→撞墙(留s2,r=-1)
    P2 = np.zeros((2, 3, 2))
    R2 = np.zeros((2, 3))
    P2[0, 0, 0] = 1; R2[0, 0] = -1
    P2[0, 1, 0] = 1; R2[0, 1] = 0
    P2[0, 2, 1] = 1; R2[0, 2] = 1
    P2[1, 0, 0] = 1; R2[1, 0] = 0
    P2[1, 1, 1] = 1; R2[1, 1] = 1
    P2[1, 2, 1] = 1; R2[1, 2] = -1
    gamma2 = 0.9

    def eval_iters(pi, n):
        Ppi = np.einsum("sa,sat->st", pi, P2)
        rpi = np.einsum("sa,sa->s", pi, R2)
        v = np.zeros(2)
        hist = [v.copy()]
        for _ in range(n):
            v = rpi + gamma2 * Ppi.dot(v)
            hist.append(v.copy())
        return hist

    pi0_2 = np.zeros((2, 3)); pi0_2[0, 0] = 1; pi0_2[1, 0] = 1   # 都向左
    return P2, R2, eval_iters, gamma2, pi0_2


@app.cell
def _(mo):
    j_slider = mo.ui.slider(0, 30, step=1, value=0,
                            label="策略评估迭代次数 j", show_value=True,
                            full_width=True)
    j_slider
    return (j_slider,)


@app.cell
def _(P2, R2, eval_iters, gamma2, j_slider, mo, np, pi0_2, plt):
    _hist = eval_iters(pi0_2, 30)
    _j = j_slider.value
    _vj = _hist[_j]
    # 收敛后做一步策略改进
    _vstar0 = _hist[-1]
    _q = R2 + gamma2 * P2.dot(_vstar0)
    _improved = _q.argmax(1)   # s1->2(右), s2->1(停)
    _anames = ["左", "停", "右"]
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _xs = range(len(_hist))
    _ax.plot(_xs, [h[0] for h in _hist], "-o", ms=3, label="v(s1)")
    _ax.plot(_xs, [h[1] for h in _hist], "-s", ms=3, label="v(s2)")
    _ax.axhline(-10, color="gray", ls="--", lw=0.8)
    _ax.axhline(-9, color="gray", ls="--", lw=0.8)
    _ax.axvline(_j, color="red", lw=1)
    _ax.set_xlabel("评估迭代 j"); _ax.set_ylabel("v")
    _ax.set_title("策略评估迭代收敛到 vπ0=(−10,−9)")
    _ax.legend(); _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""**j={_j} 时**：v(s1)={_vj[0]:.3f}, v(s2)={_vj[1]:.3f}
（书中 j=1,2,3 → s1: −1, −1.9, −2.71；最终 → −10, −9）

**策略改进**（用收敛后的 vπ0 算 q）：

| 状态 | aℓ左 | a0停 | ar右 | 改进动作 |
|---|---|---|---|---|
| s1 | {_q[0,0]:.1f} | {_q[0,1]:.1f} | {_q[0,2]:.1f} | **{_anames[_improved[0]]}** |
| s2 | {_q[1,0]:.1f} | {_q[1,1]:.1f} | {_q[1,2]:.1f} | **{_anames[_improved[1]]}** |

→ π1：s1 向右、s2 停。**一步即最优**。""")
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · 策略迭代演化 (Fig 4.4) — 🎛️ 逐步回放
    经典 5×5 网格，但 $r_{\text{forbidden}}=-10$（重罚禁区）。策略迭代 = 反复
    **完整策略评估 + 策略改进**。**拖动迭代轮 $k$** 观察书中两个现象：

    1. **离目标近的状态先找到最优策略**，远处的随后跟上；
    2. 离目标越近，状态值越大。

    最终 $\pi_k$ 收敛到与 Fig 3.4(d) 相同的最优策略与状态值。
    """)
    return


@app.cell
def _(GridWorld, greedy_policy_from_q, np, q_from_v):
    def policy_iteration_snapshots(env, gamma, max_k=30):
        P, R = env.build_model()
        nS, nA = env.n_states, env.n_actions
        # 固定的初始策略：一律向上（确定、可复现）
        pi = np.zeros((nS, nA)); pi[:, 0] = 1.0
        snaps = []
        for _ in range(max_k):
            Ppi = np.einsum("sa,sat->st", pi, P)
            rpi = np.einsum("sa,sa->s", pi, R)
            v = np.linalg.solve(np.eye(nS) - gamma * Ppi, rpi)
            snaps.append((pi.copy(), v.copy()))
            pi_new = greedy_policy_from_q(q_from_v(v, P, R, gamma))
            if np.array_equal(pi_new.argmax(1), pi.argmax(1)):
                snaps.append((pi_new.copy(), v.copy()))
                break
            pi = pi_new
        return snaps

    env_pi = GridWorld(
        cols=5, rows=5, start=(0, 0), target=(2, 3),
        forbidden=((1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)),
        r_target=1.0, r_forbidden=-10.0, r_step=0.0, r_boundary=-1.0, gamma=0.9)
    pi_snaps = policy_iteration_snapshots(env_pi, env_pi.gamma)
    return env_pi, pi_snaps


@app.cell
def _(mo, pi_snaps):
    pk_slider = mo.ui.slider(0, len(pi_snaps) - 1, step=1, value=0,
                             label="策略迭代轮 k", show_value=True, full_width=True)
    pk_slider
    return (pk_slider,)


@app.cell
def _(env_pi, pi_snaps, pk_slider):
    _k = min(pk_slider.value, len(pi_snaps) - 1)
    _pi, _v = pi_snaps[_k]
    env_pi.plot_policy(_pi, _v, precision=1,
                       title=f"策略迭代 @ k={_k}（共 {len(pi_snaps)-1} 轮收敛）")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 4 · 值迭代 vs 策略迭代 vs 截断策略迭代 (Fig 4.5)
    三者的关系：**截断策略迭代**在策略评估步只迭代 $j_{\text{truncate}}$ 次。
    - $j_{\text{truncate}}=1$ → 退化成**值迭代**（收敛慢）；
    - $j_{\text{truncate}}=\infty$ → 就是**策略迭代**（每轮评估到底，收敛快）；
    - 中间值则居于两者之间。

    下图画三种算法的误差 $\|v_k-v^*\|$ 随外层迭代 $k$ 的下降。**拖 $j_{\text{truncate}}$** 看截断曲线在两条基准之间移动。
    """)
    return


@app.cell
def _(mo):
    jtrunc_slider = mo.ui.slider(1, 30, step=1, value=3,
                                 label="截断评估步数 j_truncate", show_value=True,
                                 full_width=True)
    jtrunc_slider
    return (jtrunc_slider,)


@app.cell
def _(classic_example, greedy_policy_from_q, jtrunc_slider, mo, np, plt, q_from_v):
    env5 = classic_example()
    _P, _R = env5.build_model()
    _g = env5.gamma
    nS, nA = env5.n_states, env5.n_actions
    # 真·最优状态值
    _vstar = np.zeros(nS)
    for _ in range(2000):
        _vstar = (_R + _g * _P.dot(_vstar)).max(1)

    def truncated_pi(jt, outer=40):
        pi = np.ones((nS, nA)) / nA
        v = np.zeros(nS)
        errs = [np.max(np.abs(v - _vstar))]
        for _ in range(outer):
            Ppi = np.einsum("sa,sat->st", pi, _P)
            rpi = np.einsum("sa,sa->s", pi, _R)
            for _j in range(jt):
                v = rpi + _g * Ppi.dot(v)
            pi = greedy_policy_from_q(q_from_v(v, _P, _R, _g))
            errs.append(np.max(np.abs(v - _vstar)))
        return errs

    _err_vi = truncated_pi(1)           # 值迭代
    _err_pi = truncated_pi(1000)        # 策略迭代（评估到收敛）
    _err_tr = truncated_pi(jtrunc_slider.value)
    _fig, _ax = plt.subplots(figsize=(5.6, 3.4))
    _ax.semilogy(_err_vi, "-", color="#888", label="值迭代 (j=1)")
    _ax.semilogy(_err_pi, "-", color="#1976d2", label="策略迭代 (j=∞)")
    _ax.semilogy(_err_tr, "-o", ms=3, color="#e53935",
                 label=f"截断 PI (j={jtrunc_slider.value})")
    _ax.set_xlabel("外层迭代 k"); _ax.set_ylabel("‖v_k − v*‖ (log)")
    _ax.set_title("三种算法的收敛速度")
    _ax.legend(); _ax.set_xlim(0, 20); _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""**当前 j_truncate = {jtrunc_slider.value}**

- j=1：等价**值迭代**（灰），每轮只评估一次，外层迭代多。
- j=∞：**策略迭代**（蓝），每轮评估到底，外层迭代最少。
- 红线随 j 增大向蓝线靠拢、随 j 减小向灰线靠拢。

三者最终都收敛到同一个最优 $v^*$。""")
    ], widths=[3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结
    - **值迭代**：每步用 q-表贪心更新 v 和 π，不显式维护策略评估（Fig 4.2）。
    - **策略迭代**：交替"完整策略评估 + 策略改进"，评估需解线性方程/迭代到收敛（Fig 4.3/4.4）。
    - **截断策略迭代**：评估只跑有限步，是前两者的统一推广（Fig 4.5）。
    - 共同现象：离目标近的状态先收敛、状态值更大。
    下一章起进入**无模型**方法：不再用 P、R，而是从采样轨迹中学习。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
