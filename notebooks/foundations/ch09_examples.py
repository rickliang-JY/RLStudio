import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 9 章 · 书中例子复现 (Worked Examples)

    本 notebook 把赵世钰《强化学习的数学原理》第 9 章的**核心思想**逐一复现并做成
    **可交互**的。第 9 章第一次抛开"先学值、再贪心"的套路，直接把策略
    **参数化**为 $\pi(a\mid s;\theta)$，对目标 $J(\theta)$ 做**梯度上升**。

    | 例子 | 对应内容 | 演示什么 |
    |---|---|---|
    | 1. softmax 策略参数化 | §9.2 | 改 $\theta$ 看 $\pi$ 怎样变；$\nabla_\theta\log\pi=e_a-\pi$ |
    | 2. 单状态梯度上升 | §9.3 策略梯度定理 | 固定 $q$，梯度上升把概率推向最优动作 |
    | 3. REINFORCE 跑网格 | §9.4 | 蒙特卡洛策略梯度，策略逐步变"尖" |
    | 4. 基线降方差 | §9.5 | 减基线 $b(s)$ 不改期望、却大幅降梯度方差 |

    **策略梯度定理：**
    $$ \nabla_\theta J(\theta)=\mathbb{E}_{s\sim d_\pi,\,a\sim\pi}
       \big[\nabla_\theta\log\pi(a|s;\theta)\,q_\pi(s,a)\big]. $$
    用 softmax 参数化 $\pi(a|s)=\dfrac{e^{\theta_{s,a}}}{\sum_{a'}e^{\theta_{s,a'}}}$ 时，
    $\nabla_{\theta_s}\log\pi(a|s)=e_a-\pi(\cdot|s)$。

    > 第 3、4 例的 5×5 网格（episodic：进入目标即结束）奖励设定：
    > $r_{\text{target}}=1,\ r_{\text{forbidden}}=-1,\ r_{\text{boundary}}=-1,\ r_{\text{step}}=-0.1,\ \gamma=0.9$。
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from gridworld import GridWorld, greedy_policy_from_q
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    def book_env():
        return GridWorld(
            cols=5, rows=5, start=(0, 0), target=(2, 3),
            forbidden=((1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)),
            r_target=1.0, r_forbidden=-1.0, r_step=-0.1,
            r_boundary=-1.0, gamma=0.9,
        )

    env = book_env()
    gamma = env.gamma
    return env, gamma, greedy_policy_from_q, np, plt


@app.cell
def _(np):
    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def policy_from_theta(theta):
        z = theta - theta.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    return policy_from_theta, softmax


@app.cell
def _(env, gamma, greedy_policy_from_q, np):
    # episodic（进入目标即终止）下的最优策略——作为第 3 例 REINFORCE 的对照
    def episodic_optimal(_env, _gamma, tol=1e-12):
        _P, _R = _env.build_model()
        _tgt = _env.s_of(*_env.target)
        _n, _nA = _env.n_states, _env.n_actions
        _v = np.zeros(_n)
        while True:
            _Q = np.zeros((_n, _nA))
            for _s in range(_n):
                for _a in range(_nA):
                    _ns = int(_P[_s, _a].argmax())
                    _Q[_s, _a] = _R[_s, _a] + (0.0 if _ns == _tgt else _gamma * _v[_ns])
            _vn = _Q.max(1)
            _vn[_tgt] = 0.0
            if np.max(np.abs(_vn - _v)) < tol:
                return _vn, _Q
            _v = _vn

    v_ep, q_ep = episodic_optimal(env, gamma)
    pi_opt = greedy_policy_from_q(q_ep)
    return (pi_opt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · softmax 策略参数化 (§9.2)

    单个状态有 5 个动作（上/右/下/左/不动）。把每个动作的参数
    $\theta_a$ 调一调，看 softmax 概率 $\pi(\cdot)$ 怎样变化：
    **$\theta_a$ 越大，对应动作概率越高**。右图画出对数似然梯度
    $\nabla_{\theta}\log\pi(a\!=\!\text{所选})=e_a-\pi$ —— 这正是
    REINFORCE 更新里乘在回报前面的那一项：**所选动作分量为正（被推高），
    其余为负（被压低）**。
    """)
    return


@app.cell
def _(mo):
    th_up = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.5, label="θ(上)", show_value=True)
    th_right = mo.ui.slider(-3.0, 3.0, step=0.1, value=1.0, label="θ(右)", show_value=True)
    th_down = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.0, label="θ(下)", show_value=True)
    th_left = mo.ui.slider(-3.0, 3.0, step=0.1, value=-1.0, label="θ(左)", show_value=True)
    th_stay = mo.ui.slider(-3.0, 3.0, step=0.1, value=0.0, label="θ(不动)", show_value=True)
    chosen = mo.ui.dropdown(
        options={"上": 0, "右": 1, "下": 2, "左": 3, "不动": 4},
        value="右", label="求 ∇logπ 的所选动作 a")
    mo.vstack([th_up, th_right, th_down, th_left, th_stay, chosen])
    return chosen, th_down, th_left, th_right, th_stay, th_up


@app.cell
def _(chosen, np, plt, softmax, th_down, th_left, th_right, th_stay, th_up):
    _theta = np.array([th_up.value, th_right.value, th_down.value,
                       th_left.value, th_stay.value])
    _pi = softmax(_theta)
    _a = chosen.value
    _grad = -_pi.copy()
    _grad[_a] += 1.0  # e_a - pi
    _names = ["上", "右", "下", "左", "不动"]

    _fig1, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(8.4, 3.4))
    _bars = _ax1.bar(_names, _pi, color="#4f9fd8")
    _bars[_a].set_color("#d8754f")
    _ax1.set_ylim(0, 1)
    _ax1.set_ylabel("π(a)")
    _ax1.set_title("softmax 策略概率")
    for _i, _v in enumerate(_pi):
        _ax1.text(_i, _v + 0.02, f"{_v:.2f}", ha="center", fontsize=8)

    _cols = ["#d8754f" if _g >= 0 else "#5fa85f" for _g in _grad]
    _ax2.bar(_names, _grad, color=_cols)
    _ax2.axhline(0, color="gray", linewidth=0.8)
    _ax2.set_title(f"对数似然梯度 e_a - π（所选 a={_names[_a]}）")
    _ax2.set_ylabel("梯度分量")
    for _i, _g in enumerate(_grad):
        _ax2.text(_i, _g + (0.02 if _g >= 0 else -0.05), f"{_g:+.2f}",
                  ha="center", fontsize=8)
    _fig1.tight_layout()
    _fig1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · 单状态梯度上升 (§9.3)

    把策略梯度定理用在**一个状态**上：动作值 $q(a)$ 已知且固定，
    目标 $J(\theta)=\sum_a \pi(a)\,q(a)=\mathbb{E}_{a\sim\pi}[q(a)]$。
    其梯度恰为
    $$ \nabla_\theta J=\sum_a \pi(a)\,(e_a-\pi)\,q(a)
       =\mathbb{E}_{a\sim\pi}\big[(e_a-\pi)\,q(a)\big]. $$
    沿梯度上升，概率会**逐渐集中到 $q$ 最大的动作**上。下面给定一组
    $q$，拖动滑块看迭代多少步，策略如何从均匀走向（近似）确定性，
    $J$ 单调上升到 $\max_a q(a)$。
    """)
    return


@app.cell
def _(mo):
    pg_lr = mo.ui.slider(0.05, 1.0, step=0.05, value=0.3,
                         label="学习率 α", show_value=True, full_width=True)
    pg_steps = mo.ui.slider(0, 200, step=1, value=0,
                            label="梯度上升步数", show_value=True, full_width=True)
    mo.vstack([pg_lr, pg_steps])
    return pg_lr, pg_steps


@app.cell
def _(np, pg_lr, pg_steps, plt, softmax):
    _q = np.array([2.0, 0.5, -1.0, 0.0, 1.0])  # 固定的动作值
    _names = ["上", "右", "下", "左", "不动"]
    _theta = np.zeros(5)
    _Js = [float(softmax(_theta) @ _q)]
    for _ in range(pg_steps.value):
        _pi = softmax(_theta)
        # ∇J = Σ_a π(a)(e_a-π) q(a)
        _grad = np.zeros(5)
        for _a in range(5):
            _gl = -_pi.copy()
            _gl[_a] += 1.0
            _grad += _pi[_a] * _gl * _q[_a]
        _theta = _theta + pg_lr.value * _grad
        _Js.append(float(softmax(_theta) @ _q))
    _pi = softmax(_theta)
    _best = int(_q.argmax())

    _fig2, (_axA, _axB) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    _bars = _axA.bar(_names, _pi, color="#4f9fd8")
    _bars[_best].set_color("#d8754f")
    _axA.set_ylim(0, 1)
    _axA.set_title(f"第 {pg_steps.value} 步的 π（q={list(_q)}）")
    for _i, _v in enumerate(_pi):
        _axA.text(_i, _v + 0.02, f"{_v:.2f}", ha="center", fontsize=8)
    _axB.plot(_Js, "-o", markersize=3, color="#d8754f")
    _axB.axhline(_q.max(), color="gray", ls="--", lw=1, label=f"max q = {_q.max():.1f}")
    _axB.set_xlabel("梯度上升步")
    _axB.set_ylabel("J(θ) = E[q]")
    _axB.set_title("目标单调上升")
    _axB.legend(fontsize=8)
    _axB.grid(True, alpha=0.3)
    _fig2.tight_layout()
    _fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · REINFORCE 跑网格 (§9.4) — 🎛️ 逐步回放

    把整条 episode 的**真实回报** $G_t$ 当作 $q_\pi(s_t,a_t)$ 的无偏估计：
    $$ \theta\leftarrow\theta+\alpha\,\gamma^t\,(G_t-b(s_t))\,\nabla_\theta\log\pi(a_t|s_t). $$
    基线 $b(s)\approx\hat v(s)$ 在线学习。**探索性起点**保证每个格子都被更新。
    每隔若干 episode 存一张快照，**拖动滑块**看随机策略逐渐变"尖"、贴近 $\pi^\*$。
    """)
    return


@app.cell
def _(env, gamma, np, policy_from_theta, softmax):
    def reinforce(_env, _gamma, alpha_theta=0.05, alpha_v=0.1,
                  episodes=5000, max_steps=80, seed=1, n_snaps=40, baseline=True):
        _rng = np.random.default_rng(seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _theta = np.zeros((_nS, _nA))
        _v = np.zeros(_nS)
        _returns = []
        _snaps = []
        _snap_at = set(int(x) for x in
                       np.linspace(0, episodes, n_snaps, endpoint=False))
        for _ep in range(episodes):
            if _ep in _snap_at:
                _snaps.append((_ep, policy_from_theta(_theta), _v.copy()))
            _s = _env.reset(int(_rng.integers(_nS)))
            _traj = []
            for _ in range(max_steps):
                _p = softmax(_theta[_s])
                _a = _rng.choice(_nA, p=_p)
                _ns, _r, _done, _ = _env.step(_a)
                _traj.append((_s, _a, _r))
                _s = _ns
                if _done:
                    break
            _G = 0.0
            _Gs = []
            for (_s, _a, _r) in reversed(_traj):
                _G = _r + _gamma * _G
                _Gs.append(_G)
            _Gs.reverse()
            _returns.append(_Gs[0] if _Gs else 0.0)
            for _t, (_s, _a, _r) in enumerate(_traj):
                _adv = _Gs[_t] - (_v[_s] if baseline else 0.0)
                _v[_s] += alpha_v * (_Gs[_t] - _v[_s])
                _p = softmax(_theta[_s])
                _gl = -_p
                _gl[_a] += 1.0
                _theta[_s] += alpha_theta * (_gamma ** _t) * _adv * _gl
        _snaps.append((episodes, policy_from_theta(_theta), _v.copy()))
        return _theta, _returns, _snaps

    _theta_pg, pg_returns, pg_snaps = reinforce(env, gamma, episodes=5000,
                                                seed=1, n_snaps=40)
    return pg_returns, pg_snaps, reinforce


@app.cell
def _(mo, pg_snaps):
    pg_replay = mo.ui.slider(0, len(pg_snaps) - 1, step=1, value=0,
                             label="训练快照（越往右训练越久）", show_value=True,
                             full_width=True)
    pg_replay
    return (pg_replay,)


@app.cell
def _(env, pg_replay, pg_snaps, pi_opt):
    _idx = min(pg_replay.value, len(pg_snaps) - 1)
    _ep_at, _pi_pg, _v_base = pg_snaps[_idx]
    _agree = (_pi_pg.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(
        _pi_pg, _v_base,
        title=f"REINFORCE @ {_ep_at} episodes（与 π* 一致率 {_agree:.0%}）")
    return


@app.cell
def _(np, pg_returns, plt):
    _fig3, _ax3 = plt.subplots(figsize=(6, 3))
    _rr = np.convolve(pg_returns, np.ones(50) / 50, mode="valid")
    _ax3.plot(_rr, color="#d8754f")
    _ax3.set_xlabel("episode")
    _ax3.set_ylabel("每回合折扣回报 (滑动平均)")
    _ax3.set_title("REINFORCE 学习曲线")
    _ax3.grid(True, alpha=0.3)
    _fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    策略梯度学到的是**随机策略**（同一格可能有多个箭头），随训练逐渐变"尖"
    （接近确定性）。基线值 $\hat v(s)$ 显示在格子上方。

    ## 例子 4 · 基线为何降方差 (§9.5)

    在回报里**减去一个只依赖状态的基线** $b(s)$ 不改变梯度期望
    （因为 $\mathbb{E}_a[(e_a-\pi)\,b]=b\sum_a(\pi-\pi)\cdot\ldots=0$），
    却能**大幅降低方差**。直观看：若所有回报都被抬高一个大常数，
    $G\,\nabla\log\pi$ 会被这个大数整体放大、抖得厉害；而
    $(G-b)\,\nabla\log\pi$ 把公共部分扣掉，只留下"动作相对好坏"。

    下面在**单状态**上做蒙特卡洛：固定 softmax 策略，回报
    $G=q(a)+\text{噪声}$，整体加一个**奖励偏移**。拖动偏移，对比
    **无基线**与**用 $b=\hat v=\mathbb{E}_\pi[q]$** 时，梯度估计的总方差。
    """)
    return


@app.cell
def _(mo):
    base_offset = mo.ui.slider(0.0, 50.0, step=1.0, value=0.0,
                               label="奖励整体偏移量", show_value=True,
                               full_width=True)
    base_offset
    return (base_offset,)


@app.cell
def _(base_offset, np, plt, softmax):
    _q0 = np.array([2.0, 0.5, -1.0, 0.0, 1.0])
    _theta = np.array([0.3, 0.0, -0.2, 0.1, 0.2])
    _pi = softmax(_theta)
    _q = _q0 + base_offset.value
    _b = float(_pi @ _q)  # 基线 = 状态值
    _rng = np.random.default_rng(0)
    _N = 8000
    _g_no = np.zeros((_N, 5))
    _g_base = np.zeros((_N, 5))
    for _i in range(_N):
        _a = _rng.choice(5, p=_pi)
        _G = _q[_a] + _rng.normal(0.0, 0.5)
        _gl = -_pi.copy()
        _gl[_a] += 1.0
        _g_no[_i] = _G * _gl
        _g_base[_i] = (_G - _b) * _gl
    _var_no = _g_no.var(axis=0).sum()
    _var_base = _g_base.var(axis=0).sum()
    _reduce = 100 * (1 - _var_base / _var_no)

    _fig4, _ax4 = plt.subplots(figsize=(5.6, 3.4))
    _bars = _ax4.bar(["无基线", "有基线 b=v"], [_var_no, _var_base],
                     color=["#5fa85f", "#d8754f"])
    _ax4.set_ylabel("梯度估计总方差")
    _ax4.set_title(f"偏移={base_offset.value:.0f} 时方差降低 {_reduce:.0f}%")
    for _bar, _val in zip(_bars, [_var_no, _var_base]):
        _ax4.text(_bar.get_x() + _bar.get_width() / 2,
                  _val, f"{_val:.2f}", ha="center", va="bottom", fontsize=9)
    _fig4.tight_layout()
    _fig4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    把偏移调大，**无基线**的方差爆炸式增长，**有基线**几乎不变 ——
    这正是基线（乃至后面 Actor-Critic 里的 critic）存在的意义：
    扣掉与动作无关的公共项，只让"动作相对优势"驱动更新。

    ## 小结
    - 策略梯度**直接优化参数化策略**，天然输出随机策略，适合大/连续动作空间。
    - softmax 参数化下 $\nabla_\theta\log\pi=e_a-\pi$：所选动作被推高、其余被压低。
    - REINFORCE 用整条 episode 的回报 $G_t$ 作 $q_\pi$ 的无偏估计 → 方差大。
    - **基线** $b(s)$ 不改期望、却显著降方差；把它升级成**自举的 critic**，
      就是下一章的 **Actor-Critic**：actor 出动作，critic 在线评估。
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
