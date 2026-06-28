import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 7 章 · 书中例子复现 (Worked Examples)

    本notebook把赵世钰《强化学习的数学原理》第 7 章里出现的**经典插图例子**逐一
    复现出来，并全部做成**可交互**的：拖动滑块/换参数，立刻看到书里那张图怎么变。

    | 例子 | 书中图 | 演示什么 |
    |---|---|---|
    | 1. TD 估计状态值 | Fig 7.3 | 单条轨迹在线更新，估计误差怎样压到 0 |
    | 2. Sarsa 找最优路径 | Fig 7.4 | 固定起点出发，回报曲线上升、步数曲线下降 |
    | 3. Q-learning 离线学最优 | Fig 7.6/7.7 | 用**探索性行为策略**的数据，直接学出 $q^*$ |
    | 4. 行为策略探索性的影响 | Fig 7.8 | 行为策略探索越充分，离线 Q-learning 越能找全最优 |

    > 第 7 章 5×5 例子的奖励设定（与本notebook一致）：
    > $r_{\text{boundary}}=-1,\ r_{\text{forbidden}}=-10,\ r_{\text{target}}=1,\ r_{\text{other}}=0,\ \gamma=0.9$。

    > **统一公式**：新估计 ← 旧估计 + α·[ TD target − 旧估计 ]。
    > TD = 蒙特卡洛的在线版 + 随机近似的应用——不必等 episode 结束，每走一步就用
    > **自举 (bootstrapping)** 更新。
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from gridworld import GridWorld, greedy_policy_from_q, q_from_v
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return GridWorld, greedy_policy_from_q, np, plt, q_from_v


@app.cell
def _(GridWorld):
    # 第 7 章 5×5 例子的网格：r_forbidden=-10（比前几章更狠，正是书里这组数值的来源）
    def book_grid():
        return GridWorld(
            cols=5, rows=5, start=(0, 0), target=(2, 3),
            forbidden=((1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)),
            r_target=1.0, r_forbidden=-10.0, r_step=0.0,
            r_boundary=-1.0, gamma=0.9,
        )
    env = book_grid()
    gamma = env.gamma
    return book_grid, env, gamma


@app.cell
def _(env, gamma, greedy_policy_from_q, np, q_from_v):
    # 真值（地面真相）：值迭代求 v*、q*、π*，作为后面所有 TD 估计的对照
    def solve_v(_env, _pi, _gamma):
        _P, _R = _env.build_model()
        _Ppi = np.einsum("sa,sat->st", _pi, _P)
        _rpi = np.einsum("sa,sa->s", _pi, _R)
        return np.linalg.solve(np.eye(_env.n_states) - _gamma * _Ppi, _rpi)

    def value_iteration(_env, _gamma, tol=1e-12):
        _P, _R = _env.build_model()
        _v = np.zeros(_env.n_states)
        while True:
            _q = _R + _gamma * _P.dot(_v)
            _vn = _q.max(1)
            if np.max(np.abs(_vn - _v)) < tol:
                return _vn
            _v = _vn

    _Pm, _Rm = env.build_model()
    v_opt = value_iteration(env, gamma)
    q_opt = q_from_v(v_opt, _Pm, _Rm, gamma)
    pi_opt = greedy_policy_from_q(q_opt)
    return pi_opt, solve_v, v_opt, value_iteration


@app.cell
def _(np):
    # 公共小工具：ε-greedy 选动作、按概率采样动作、滑动平均
    def eps_greedy_action(Q, s, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(Q.shape[1]))
        return int(Q[s].argmax())

    def sample_action(pi_row, rng):
        return int(rng.choice(len(pi_row), p=pi_row))

    def smooth(x, k=50):
        x = np.asarray(x, float)
        if len(x) < k:
            return x
        return np.convolve(x, np.ones(k) / k, mode="valid")

    return eps_greedy_action, sample_action, smooth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · TD 估计状态值 (Fig 7.3)
    给定一个策略 $\pi$（这里取最优策略 $\pi^*$），**只用采样**来估计它的状态值
    $v_\pi$——不需要模型、不需要等 episode 结束。每走一步就更新：

    $$v(s_t)\leftarrow v(s_t)+\alpha\big[\,r_{t+1}+\gamma v(s_{t+1})-v(s_t)\,\big].$$

    左图是**估计误差**（与真值 $v_\pi$ 的 RMS 误差）随步数下降；右图是收敛后的
    估计值，应当与真值逐格吻合。拖动 α 和步数看收敛快慢。
    """)
    return


@app.cell
def _(mo):
    alpha1_dropdown = mo.ui.dropdown(
        {"0.1": 0.1, "0.05": 0.05, "0.2": 0.2, "0.5": 0.5},
        value="0.1", label="学习率 α（常数步长）")
    steps1_slider = mo.ui.slider(2000, 40000, step=2000, value=20000,
                                 label="采样步数", show_value=True, full_width=True)
    seed1_slider = mo.ui.slider(0, 20, step=1, value=0,
                                label="随机种子", show_value=True, full_width=True)
    mo.vstack([alpha1_dropdown, steps1_slider, seed1_slider])
    return alpha1_dropdown, seed1_slider, steps1_slider


@app.cell
def _(
    alpha1_dropdown,
    env,
    gamma,
    mo,
    np,
    pi_opt,
    plt,
    sample_action,
    seed1_slider,
    steps1_slider,
    v_opt,
):
    def td_state_value(_env, _pi, _gamma, _true_v, _alpha, _n_steps, _seed):
        _rng = np.random.default_rng(_seed)
        _v = np.zeros(_env.n_states)
        _s = _env.reset(int(_rng.integers(_env.n_states)))
        _errs = []
        for _t in range(_n_steps):
            _a = sample_action(_pi[_s], _rng)
            _ns, _r, _done, _ = _env.step(_a)
            _v[_s] += _alpha * (_r + _gamma * _v[_ns] - _v[_s])
            _errs.append(float(np.sqrt(np.mean((_v - _true_v) ** 2))))
            _s = _env.reset(int(_rng.integers(_env.n_states))) if _done else _ns
        return _v, _errs

    _v_est, _errs = td_state_value(env, pi_opt, gamma, v_opt,
                                   alpha1_dropdown.value, steps1_slider.value,
                                   seed1_slider.value)
    _fig, _ax = plt.subplots(figsize=(5.4, 3.4))
    _ax.plot(_errs, color="#e53935", lw=1.0)
    _ax.set_xlabel("采样步数 t"); _ax.set_ylabel("RMS 误差 ‖v−v_pi‖")
    _ax.set_title(f"TD 估计误差（α={alpha1_dropdown.value}）")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    _grid = env.plot_policy(pi_opt, _v_est, title="收敛后的 TD 估计值",
                            precision=1, figsize=(4.4, 4.4))
    _maxerr = float(np.max(np.abs(_v_est - v_opt)))
    mo.hstack([
        _fig, _grid,
        mo.md(f"""**最终** RMS 误差 = **{_errs[-1]:.3f}**，
逐格最大误差 = **{_maxerr:.3f}**。

- α 越大前期下降越快，但残余抖动也越大（常数步长不满足
  $\\sum a_k^2<\\infty$，所以收不到 0，只在真值附近抖动）；
- 步数越多覆盖越充分，所有格子才都被估准。""")
    ], widths=[3, 3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · Sarsa 找最优路径 (Fig 7.4)
    任务：从**固定起点**（左上角 s1）出发，用 Sarsa 一边走一边学，最终找到一条
    通往目标、绕开禁区的好路径。Sarsa 是 **on-policy** 的，TD target 用的是
    **实际要走的下一个动作** $a_{t+1}$：

    $$q(s_t,a_t)\leftarrow q(s_t,a_t)+\alpha\big[\,r_{t+1}+\gamma q(s_{t+1},a_{t+1})-q(s_t,a_t)\,\big].$$

    书里这张图有三块：**学到的策略** + **每个 episode 的总回报（上升）** +
    **每个 episode 的步数（下降）**。下面先用书中默认参数 (α=0.1, ε=0.1) 跑出来。
    """)
    return


@app.cell
def _(eps_greedy_action, np):
    def sarsa(env, gamma, alpha, eps, episodes, start_state,
              max_steps=1000, seed=0):
        rng = np.random.default_rng(seed)
        Q = np.zeros((env.n_states, env.n_actions))
        rets, lens = [], []
        for _ep in range(episodes):
            _s = env.reset(start_state)
            _a = eps_greedy_action(Q, _s, eps, rng)
            _G = 0.0
            for _t in range(max_steps):
                _ns, _r, _done, _ = env.step(_a)
                _na = eps_greedy_action(Q, _ns, eps, rng)
                Q[_s, _a] += alpha * (_r + gamma * Q[_ns, _na] - Q[_s, _a])
                _s, _a = _ns, _na
                _G += _r
                if _done:
                    break
            rets.append(_G)
            lens.append(_t + 1)
        return Q, rets, lens

    return (sarsa,)


@app.cell
def _(
    env,
    gamma,
    greedy_policy_from_q,
    mo,
    pi_opt,
    plt,
    sarsa,
    smooth,
):
    _start = env.s_of(0, 0)
    _Q, _rets, _lens = sarsa(env, gamma, alpha=0.1, eps=0.1, episodes=500,
                             start_state=_start, seed=1)
    _pi = greedy_policy_from_q(_Q)
    _agree = (_pi.argmax(1) == pi_opt.argmax(1)).mean()
    _grid = env.plot_policy(_pi, _Q.max(1),
                            title=f"Sarsa 学到的策略 (与 π* 一致 {_agree:.0%})",
                            figsize=(4.4, 4.4))

    _fig, _axs = plt.subplots(2, 1, figsize=(5.4, 4.2), sharex=True)
    _axs[0].plot(_rets, color="#bbb", lw=0.6)
    _axs[0].plot(smooth(_rets), color="#1976d2", lw=1.4)
    _axs[0].set_ylabel("每回合总回报")
    _axs[0].set_title("Sarsa 学习曲线（默认 α=0.1, ε=0.1）")
    _axs[0].grid(True, alpha=0.3)
    _axs[1].plot(_lens, color="#bbb", lw=0.6)
    _axs[1].plot(smooth(_lens), color="#e53935", lw=1.4)
    _axs[1].set_ylabel("每回合步数"); _axs[1].set_xlabel("episode")
    _axs[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.hstack([_grid, _fig], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    回报曲线从很负（一头撞进禁区拿 −10）一路爬升并趋于平稳，步数曲线则从几百步
    猛降到十几步——这正是"策略变好"的两个侧面。下面**亲手调参**再看一次：
    """)
    return


@app.cell
def _(mo):
    alpha2_slider = mo.ui.slider(0.01, 0.9, step=0.01, value=0.1,
                                 label="学习率 α", show_value=True, full_width=True)
    eps2_slider = mo.ui.slider(0.0, 0.6, step=0.02, value=0.1,
                               label="探索率 ε", show_value=True, full_width=True)
    episodes2_slider = mo.ui.slider(100, 2000, step=100, value=500,
                                    label="训练 episode 数", show_value=True,
                                    full_width=True)
    run2_btn = mo.ui.run_button(label="▶ 用以上参数重新训练 Sarsa")
    mo.vstack([alpha2_slider, eps2_slider, episodes2_slider, run2_btn])
    return alpha2_slider, episodes2_slider, eps2_slider, run2_btn


@app.cell
def _(
    alpha2_slider,
    env,
    episodes2_slider,
    eps2_slider,
    gamma,
    greedy_policy_from_q,
    mo,
    pi_opt,
    plt,
    run2_btn,
    sarsa,
    smooth,
):
    mo.stop(not run2_btn.value,
            mo.md("⬆️ **调好上面的滑块后，点击「▶ 重新训练」按钮**，这里会显示结果。"))
    _start = env.s_of(0, 0)
    _Q, _rets, _lens = sarsa(env, gamma, alpha=alpha2_slider.value,
                             eps=eps2_slider.value,
                             episodes=episodes2_slider.value,
                             start_state=_start, seed=7)
    _pi = greedy_policy_from_q(_Q)
    _agree = (_pi.argmax(1) == pi_opt.argmax(1)).mean()
    _grid = env.plot_policy(
        _pi, _Q.max(1),
        title=(f"α={alpha2_slider.value:.2f}, ε={eps2_slider.value:.2f} "
               f"→ 一致 {_agree:.0%}"), figsize=(4.4, 4.4))
    _fig, _axs = plt.subplots(2, 1, figsize=(5.4, 4.2), sharex=True)
    _axs[0].plot(smooth(_rets), color="#1976d2", lw=1.4)
    _axs[0].set_ylabel("每回合总回报"); _axs[0].grid(True, alpha=0.3)
    _axs[1].plot(smooth(_lens), color="#e53935", lw=1.4)
    _axs[1].set_ylabel("每回合步数"); _axs[1].set_xlabel("episode")
    _axs[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.hstack([_grid, _fig], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · Q-learning 离线学最优 (Fig 7.6 / 7.7)
    Q-learning 是 **off-policy** 的：TD target 用 $\max_{a'}q(s',a')$，直接逼近
    **最优**动作值 $q^*$，而**不在乎数据是谁生成的**。

    $$q(s_t,a_t)\leftarrow q(s_t,a_t)+\alpha\big[\,r_{t+1}+\gamma\max_{a'}q(s_{t+1},a')-q(s_t,a_t)\,\big].$$

    书里的演示：用一个**固定的、探索性很强的行为策略**（这里取均匀随机）去生成数据，
    Q-learning 只看这些数据就能学出最优策略。左图是行为策略，中图是估计值误差随步数
    下降，右图（可回放）是学到的策略一步步逼近 $\pi^*$。
    """)
    return


@app.cell
def _(np, v_opt):
    def q_learning_offpolicy(env, gamma, behavior_pi, alpha, n_steps,
                             seed=0, n_snaps=40):
        rng = np.random.default_rng(seed)
        Q = np.zeros((env.n_states, env.n_actions))
        errs, snaps = [], []
        snap_at = set(int(x) for x in
                      np.linspace(0, n_steps, n_snaps, endpoint=False))
        _s = env.reset(int(rng.integers(env.n_states)))
        for _t in range(n_steps):
            if _t in snap_at:
                snaps.append((_t, Q.copy()))
            _a = int(rng.choice(env.n_actions, p=behavior_pi[_s]))
            _ns, _r, _done, _ = env.step(_a)
            Q[_s, _a] += alpha * (_r + gamma * Q[_ns].max() - Q[_s, _a])
            errs.append(float(np.sqrt(np.mean((Q.max(1) - v_opt) ** 2))))
            _s = env.reset(int(rng.integers(env.n_states))) if _done else _ns
        snaps.append((n_steps, Q.copy()))
        return Q, errs, snaps

    return (q_learning_offpolicy,)


@app.cell
def _(env, gamma, np, q_learning_offpolicy):
    # 行为策略：均匀随机（每个动作概率 1/5），探索性最强
    behavior_uniform = np.full((env.n_states, env.n_actions), 1.0 / env.n_actions)
    _Q3, ql_errs, ql_snaps = q_learning_offpolicy(
        env, gamma, behavior_uniform, alpha=0.1, n_steps=30000, seed=2)
    return behavior_uniform, ql_errs, ql_snaps


@app.cell
def _(mo, ql_snaps):
    snap3_slider = mo.ui.slider(0, len(ql_snaps) - 1, step=1, value=0,
                                label="训练快照（越往右学得越久）", show_value=True,
                                full_width=True)
    snap3_slider
    return (snap3_slider,)


@app.cell
def _(
    behavior_uniform,
    env,
    greedy_policy_from_q,
    mo,
    pi_opt,
    plt,
    ql_errs,
    ql_snaps,
    snap3_slider,
):
    _beh_grid = env.plot_policy(behavior_uniform, title="行为策略（均匀随机）",
                                figsize=(4.0, 4.0))

    _fig, _ax = plt.subplots(figsize=(4.6, 3.0))
    _ax.plot(ql_errs, color="#e53935", lw=1.0)
    _ax.set_xlabel("更新步数"); _ax.set_ylabel("RMS 误差 ‖max_a q − v*‖")
    _ax.set_title("Q-learning 估计误差")
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    _idx = min(snap3_slider.value, len(ql_snaps) - 1)
    _ep_at, _Q_snap = ql_snaps[_idx]
    _pi_snap = greedy_policy_from_q(_Q_snap)
    _agree = (_pi_snap.argmax(1) == pi_opt.argmax(1)).mean()
    _grid = env.plot_policy(
        _pi_snap, _Q_snap.max(1),
        title=f"@ {_ep_at} 步  (与 π* 一致 {_agree:.0%})", figsize=(4.0, 4.0))
    mo.hstack([_beh_grid, _fig, _grid], widths=[1, 1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    把回放滑块从左拖到右：尽管数据全部来自**毫无目的的随机游走**，Q-learning 仍然把
    箭头一点点对齐成最优路线——这就是 off-policy 的威力（行为策略负责"探索"，
    目标策略负责"最优"）。

    ## 例子 4 · 行为策略探索性的影响 (Fig 7.8)
    上面行为策略是均匀的、探索性最强。如果行为策略**不够探索**会怎样？
    这里用一个参数 $\epsilon$ 控制行为策略：

    $$\pi_b(a|s)=(1-\epsilon)\,\pi^*(a|s)+\epsilon\cdot\tfrac{1}{5}.$$

    - $\epsilon$ 小 → 行为策略几乎只走最优路 → 很多格子**根本没被访问** → 那些格子学不对；
    - $\epsilon$ 大 → 接近均匀 → 覆盖全 → Q-learning 把整张图都学成最优。

    下面先看一条**预扫描曲线**：一致率随 $\epsilon$ 怎样上升。
    """)
    return


@app.cell
def _(
    env,
    gamma,
    greedy_policy_from_q,
    np,
    pi_opt,
    plt,
    q_learning_offpolicy,
):
    def make_behavior(eps):
        _b = (1 - eps) * pi_opt + eps / env.n_actions
        return _b / _b.sum(1, keepdims=True)

    _eps_grid = np.linspace(0.05, 1.0, 12)
    _agrees = []
    for _e in _eps_grid:
        _Q, _, _ = q_learning_offpolicy(env, gamma, make_behavior(_e),
                                        alpha=0.1, n_steps=20000, seed=3,
                                        n_snaps=2)
        _pi = greedy_policy_from_q(_Q)
        _agrees.append((_pi.argmax(1) == pi_opt.argmax(1)).mean())

    _fig, _ax = plt.subplots(figsize=(5.6, 3.2))
    _ax.plot(_eps_grid, _agrees, "-o", color="#1976d2", ms=4)
    _ax.set_xlabel("行为策略探索度 ε"); _ax.set_ylabel("与 π* 一致率")
    _ax.set_title("探索越充分，Q-learning 越能学全最优")
    _ax.set_ylim(0, 1.05); _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    _fig
    return (make_behavior,)


@app.cell
def _(mo):
    eps4_slider = mo.ui.slider(0.05, 1.0, step=0.05, value=0.1,
                               label="行为策略探索度 ε", show_value=True,
                               full_width=True)
    run4_btn = mo.ui.run_button(label="▶ 用该 ε 跑离线 Q-learning")
    mo.vstack([eps4_slider, run4_btn])
    return eps4_slider, run4_btn


@app.cell
def _(
    env,
    eps4_slider,
    gamma,
    greedy_policy_from_q,
    make_behavior,
    mo,
    np,
    pi_opt,
    q_learning_offpolicy,
    run4_btn,
):
    mo.stop(not run4_btn.value,
            mo.md("⬆️ **拖动 ε 再点按钮**，看不同探索度下学到的策略。"))
    _beh = make_behavior(eps4_slider.value)
    _Q, _, _snaps = q_learning_offpolicy(env, gamma, _beh, alpha=0.1,
                                         n_steps=25000, seed=5, n_snaps=2)
    _pi = greedy_policy_from_q(_Q)
    _agree = (_pi.argmax(1) == pi_opt.argmax(1)).mean()
    # 统计有多少格子的最优动作真正被行为策略"足够"覆盖
    _visited = int((_beh.max(1) < 0.99).sum())  # 非确定性（被探索）的状态数
    _beh_grid = env.plot_policy(_beh, title=f"行为策略 (ε={eps4_slider.value:.2f})",
                                figsize=(4.2, 4.2))
    _grid = env.plot_policy(_pi, _Q.max(1),
                            title=f"学到的策略 (与 π* 一致 {_agree:.0%})",
                            figsize=(4.2, 4.2))
    mo.hstack([
        _beh_grid, _grid,
        mo.md(f"""ε = **{eps4_slider.value:.2f}** 时，与最优策略一致率 = **{_agree:.0%}**。

- ε 偏小：行为策略几乎沿固定一条路走，离路的格子样本太少甚至为零，
  那里的 q 值学不准，箭头会乱指；
- ε 调大到接近 1：覆盖全图，一致率升到 100%。

这正是 Fig 7.8 的核心教训：**off-policy 也要靠行为策略提供足够的探索**。""")
    ], widths=[1, 1, 1.3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结
    - **TD(0)** 在线、无模型地估 $v_\pi$：每步自举更新，常数步长会在真值附近抖动。
    - **Sarsa**（on-policy）：TD target 用实走的 $a_{t+1}$，回报曲线升、步数曲线降。
    - **Q-learning**（off-policy）：TD target 用 $\max_{a'}q$，直接学 $q^*$，
      哪怕数据来自随机游走也能学出最优。
    - 但 off-policy **不等于不用探索**：行为策略必须**覆盖**所有 $(s,a)$，
      探索不足时未访问的格子学不对（Fig 7.8）。
    - 以上都是**表格型**方法；状态一多就存不下，下一章用**函数近似**替换表格。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
