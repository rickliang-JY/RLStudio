import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 8 章 · 书中例子复现 (Worked Examples)

    本notebook把赵世钰《强化学习的数学原理》第 8 章的**核心例子**逐一复现并做成
    **可交互**的。第 8 章的主题：把**表格**换成**参数化函数** $\hat v(s;w)=\phi(s)^\top w$，
    用第 6 章的随机梯度思想训练参数，从而**泛化**到没怎么见过的状态。

    | 例子 | 书中图 | 演示什么 |
    |---|---|---|
    | 1. 用特征拟合值曲面 | Fig 8.3–8.5 | 特征阶数越高，拟合越准（多项式/傅里叶） |
    | 2. TD-Linear 在线估计 | Fig 8.6/8.7 | 边采样边拟合，误差怎样压向最小二乘下界 |
    | 3. Sarsa + 线性近似做控制 | Fig 8.9 | 用傅里叶特征做控制，回报曲线上升 |
    | 4. "深度" Q-learning | Fig 8.11 | 小神经网络 + 回放 + 目标网络（纯 numpy 手写） |

    > 本章 5×5 网格（episodic：进入目标即结束）奖励设定：
    > $r_{\text{target}}=1,\ r_{\text{forbidden}}=-1,\ r_{\text{boundary}}=-1,\ r_{\text{step}}=-0.1,\ \gamma=0.9$。
    > 加一点每步小代价 $-0.1$，让"尽快到达"成为唯一最优，控制问题更干净。
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
def _(env, gamma, greedy_policy_from_q, np):
    # episodic（进入目标即终止）下的最优策略与最优值——作为后面所有拟合/控制的对照
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

    def episodic_eval(_env, _gamma, _pi):
        _P, _R = _env.build_model()
        _tgt = _env.s_of(*_env.target)
        _n, _nA = _env.n_states, _env.n_actions
        _Ppi = np.zeros((_n, _n))
        _rpi = np.zeros(_n)
        for _s in range(_n):
            for _a in range(_nA):
                _p = _pi[_s, _a]
                if _p <= 0:
                    continue
                _ns = int(_P[_s, _a].argmax())
                _rpi[_s] += _p * _R[_s, _a]
                if _ns != _tgt:
                    _Ppi[_s, _ns] += _p * _gamma
        return np.linalg.solve(np.eye(_n) - _Ppi, _rpi)

    v_ep, q_ep = episodic_optimal(env, gamma)
    pi_ep = greedy_policy_from_q(q_ep)
    return episodic_eval, pi_ep, v_ep


@app.cell
def _(np):
    # 两类特征：多项式（i+j<=order 的单项 x^i y^j）与傅里叶（cos(π(c1 x + c2 y))）
    def build_features(_env, kind, order):
        _rows = []
        for _s in range(_env.n_states):
            _x, _y = _env.xy_of(_s)
            _x = _x / (_env.cols - 1)
            _y = _y / (_env.rows - 1)
            _row = []
            if kind == "poly":
                for _i in range(order + 1):
                    for _j in range(order + 1 - _i):
                        _row.append((_x ** _i) * (_y ** _j))
            else:  # fourier
                for _c1 in range(order + 1):
                    for _c2 in range(order + 1):
                        _row.append(np.cos(np.pi * (_c1 * _x + _c2 * _y)))
            _rows.append(_row)
        return np.array(_rows, dtype=float)

    return (build_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · 用特征拟合值曲面 (Fig 8.3–8.5)
    把 25 个状态的**最优状态值** $v^*$ 看成定义在网格上的一张曲面（左图）。
    我们不再逐格存数，而是用少量**特征**线性组合去逼近它：

    $$\hat v(s;w)=\phi(s)^\top w,\qquad w^\star=\arg\min_w\sum_s\big(v^*(s)-\phi(s)^\top w\big)^2.$$

    先用**最小二乘闭式解**直接求 $w^\star$（暂不涉及 RL），看**特征阶数**与**类型**
    如何决定拟合质量：阶数越高曲面越贴合，但参数也越多。
    """)
    return


@app.cell
def _(mo):
    kind1_dropdown = mo.ui.dropdown(
        {"多项式 polynomial": "poly", "傅里叶 Fourier": "fourier"},
        value="多项式 polynomial", label="特征类型")
    order1_slider = mo.ui.slider(1, 5, step=1, value=2,
                                 label="特征阶数 order", show_value=True,
                                 full_width=True)
    mo.vstack([kind1_dropdown, order1_slider])
    return kind1_dropdown, order1_slider


@app.cell
def _(build_features, env, kind1_dropdown, mo, np, order1_slider, plt, v_ep):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _Phi = build_features(env, kind1_dropdown.value, order1_slider.value)
    _w, _, _, _ = np.linalg.lstsq(_Phi, v_ep, rcond=None)
    _v_hat = _Phi @ _w
    _rmse = float(np.sqrt(np.mean((_v_hat - v_ep) ** 2)))

    _xx, _yy = np.meshgrid(np.arange(env.cols), np.arange(env.rows))
    _fig = plt.figure(figsize=(8.4, 3.6))
    for _i, (_vals, _ttl) in enumerate([(v_ep, "真实 v* 曲面"),
                                        (_v_hat, f"拟合曲面 (d={_Phi.shape[1]} 参数)")]):
        _ax = _fig.add_subplot(1, 2, _i + 1, projection="3d")
        _ax.plot_surface(_xx, _yy, np.asarray(_vals).reshape(env.rows, env.cols),
                         cmap="viridis", edgecolor="k", linewidth=0.2)
        _ax.set_title(_ttl); _ax.set_xlabel("x"); _ax.set_ylabel("y")
        _ax.set_zlim(v_ep.min() - 0.2, v_ep.max() + 0.2)
    _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""**RMSE = {_rmse:.3f}**，参数个数 = **{_Phi.shape[1]}**（vs 表格 25 个）。

- **阶数 ↑**：表达力变强，曲面越贴合真值，RMSE 下降；
- 低阶多项式只能抓"越靠目标值越高"的大趋势，细节有偏差；
- 傅里叶基对起伏的捕捉方式不同——这就是**特征工程**。""")
    ], widths=[3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · TD-Linear 在线估计 (Fig 8.6/8.7)
    例子 1 用了真值做最小二乘；现实里没有真值。**TD-Linear** 只靠采样
    （这里沿最优策略 $\pi^*$ 采样，进入目标即终止）：

    $$w\leftarrow w+\alpha\big[r+\gamma\,\phi(s')^\top w-\phi(s)^\top w\big]\phi(s).$$

    左图是估计误差随采样步数下降，虚线是该特征族**最好也只能到**的最小二乘下界；
    右图是收敛后的拟合值。换特征类型/阶数，看 TD 收敛点如何随之改变。
    """)
    return


@app.cell
def _(mo):
    kind2_dropdown = mo.ui.dropdown(
        {"多项式 polynomial": "poly", "傅里叶 Fourier": "fourier"},
        value="多项式 polynomial", label="特征类型")
    order2_slider = mo.ui.slider(1, 4, step=1, value=2,
                                 label="特征阶数 order", show_value=True,
                                 full_width=True)
    alpha2_dropdown = mo.ui.dropdown({"0.01": 0.01, "0.005": 0.005, "0.02": 0.02},
                                     value="0.01", label="学习率 α")
    mo.vstack([kind2_dropdown, order2_slider, alpha2_dropdown])
    return alpha2_dropdown, kind2_dropdown, order2_slider


@app.cell
def _(
    alpha2_dropdown,
    build_features,
    env,
    gamma,
    kind2_dropdown,
    mo,
    np,
    order2_slider,
    pi_ep,
    plt,
    v_ep,
):
    def linear_td(_env, _Phi, _policy, _gamma, _alpha, _v_true, _steps=120000,
                  _seed=0):
        _rng = np.random.default_rng(_seed)
        _w = np.zeros(_Phi.shape[1])
        _s = _env.reset(int(_rng.integers(_env.n_states)))
        _errs = []
        for _t in range(_steps):
            _a = int(_rng.choice(_env.n_actions, p=_policy[_s]))
            _ns, _r, _done, _ = _env.step(_a)
            _boot = 0.0 if _done else _Phi[_ns] @ _w
            _w += _alpha * (_r + _gamma * _boot - _Phi[_s] @ _w) * _Phi[_s]
            _errs.append(float(np.sqrt(np.mean((_Phi @ _w - _v_true) ** 2))))
            _s = _env.reset(int(_rng.integers(_env.n_states))) if _done else _ns
        return _w, _errs

    _Phi = build_features(env, kind2_dropdown.value, order2_slider.value)
    _wls, _, _, _ = np.linalg.lstsq(_Phi, v_ep, rcond=None)
    _floor = float(np.sqrt(np.mean((_Phi @ _wls - v_ep) ** 2)))
    _w, _errs = linear_td(env, _Phi, pi_ep, gamma, alpha2_dropdown.value, v_ep)
    _v_hat = _Phi @ _w

    _fig, _ax = plt.subplots(figsize=(5.2, 3.4))
    _ax.plot(_errs, color="#e53935", lw=1.0, label="TD-Linear 误差")
    _ax.axhline(_floor, color="gray", ls="--", lw=1.0,
                label=f"最小二乘下界 {_floor:.3f}")
    _ax.set_xlabel("采样步数"); _ax.set_ylabel("RMSE")
    _ax.set_title("TD-Linear 估计误差"); _ax.legend(fontsize=8)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    _grid = env.plot_policy(pi_ep, _v_hat, title="收敛后的拟合值",
                            precision=1, figsize=(4.2, 4.2))
    mo.hstack([
        _fig, _grid,
        mo.md(f"""TD 终值 RMSE = **{_errs[-1]:.3f}**，
该特征族下界 = **{_floor:.3f}**。

TD-Linear 最多压到那条**虚线下界**——特征族不够强（阶数低）时，
再训练也突破不了它。所以"换更好的特征"和"训练更久"是两件事。""")
    ], widths=[3, 3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · Sarsa + 线性近似做控制 (Fig 8.9)
    把近似从**值评估**推广到**控制**：动作值也参数化
    $\hat q(s,a;w)=\phi(s)^\top w_a$（每个动作一组权重）。用 Sarsa 更新：

    $$w_a\leftarrow w_a+\alpha\big[r+\gamma\,\hat q(s',a';w)-\hat q(s,a;w)\big]\phi(s).$$

    线性控制比表格更**不稳定**，需要够丰富的特征（这里用傅里叶基）+ 探索起点 +
    递减 ε。下面跑出学习曲线与最终策略。
    """)
    return


@app.cell
def _(np):
    def linear_sarsa(env, Phi, gamma, alpha, eps0, episodes, max_steps=300,
                     seed=0):
        _rng = np.random.default_rng(seed)
        _d, _nA = Phi.shape[1], env.n_actions
        W = np.zeros((_d, _nA))

        def _act(s, eps):
            if _rng.random() < eps:
                return int(_rng.integers(_nA))
            return int((Phi[s] @ W).argmax())

        rets, lens = [], []
        for _ep in range(episodes):
            _eps = max(0.05, eps0 * (1 - _ep / episodes))
            _s = env.reset(int(_rng.integers(env.n_states)))
            _a = _act(_s, _eps)
            _G = 0.0
            for _t in range(max_steps):
                _ns, _r, _done, _ = env.step(_a)
                _na = _act(_ns, _eps)
                _boot = 0.0 if _done else (Phi[_ns] @ W)[_na]
                _td = _r + gamma * _boot - (Phi[_s] @ W)[_a]
                W[:, _a] += alpha * _td * Phi[_s]
                _s, _a = _ns, _na
                _G += _r
                if _done:
                    break
            rets.append(_G)
            lens.append(_t + 1)
        return W, rets, lens

    return (linear_sarsa,)


@app.cell
def _(
    build_features,
    env,
    gamma,
    greedy_policy_from_q,
    linear_sarsa,
    mo,
    np,
    pi_ep,
    plt,
):
    _Phi = build_features(env, "fourier", 4)        # 25 个傅里叶特征
    _W, _rets, _lens = linear_sarsa(env, _Phi, gamma, alpha=0.01, eps0=0.2,
                                    episodes=2500, seed=0)
    _Q = _Phi @ _W
    _pi = greedy_policy_from_q(_Q)
    _agree = (_pi.argmax(1) == pi_ep.argmax(1)).mean()

    def _smooth(x, k=50):
        x = np.asarray(x, float)
        return x if len(x) < k else np.convolve(x, np.ones(k) / k, mode="valid")

    _fig, _axs = plt.subplots(2, 1, figsize=(5.2, 4.0), sharex=True)
    _axs[0].plot(_smooth(_rets), color="#1976d2", lw=1.4)
    _axs[0].set_ylabel("每回合回报"); _axs[0].grid(True, alpha=0.3)
    _axs[0].set_title("线性 Sarsa（25 傅里叶特征）学习曲线")
    _axs[1].plot(_smooth(_lens), color="#e53935", lw=1.4)
    _axs[1].set_ylabel("每回合步数"); _axs[1].set_xlabel("episode")
    _axs[1].grid(True, alpha=0.3)
    _fig.tight_layout()
    _grid = env.plot_policy(_pi, _Q.max(1),
                            title=f"线性 Sarsa 策略 (与 π* 一致 {_agree:.0%})",
                            figsize=(4.2, 4.2))
    mo.hstack([
        _grid, _fig,
        mo.md(f"""回报从很负（撞墙/进禁区）爬升到**正值**，步数随之下降——策略确实学好了。

与最优策略一致率 ≈ **{_agree:.0%}**：线性近似只有有限容量，
不像表格那样逐格精确，个别格子的箭头可能"差不多对但不完全对"。
这正是**泛化 vs 精确**的取舍。""")
    ], widths=[1, 1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 4 · "深度" Q-learning (Fig 8.11/8.12)
    把线性 $\phi(s)^\top w$ 换成**小神经网络** $\hat q(s,\cdot;w)$（纯 numpy 手写单隐层
    MLP，输入 one-hot 状态，输出 5 个动作的 Q 值）。两大稳定技巧：
    1. **经验回放**：把 $(s,a,r,s')$ 存入缓冲区随机采样，打破相关性；
    2. **目标网络**：用一份滞后参数 $w^-$ 算 TD target。

    损失 $L(w)=\big(r+\gamma\max_{a'}\hat q(s',a';w^-)-\hat q(s,a;w)\big)^2$。
    下图：左（可回放）为学到的策略逼近 $\pi^*$，右为学习曲线。
    """)
    return


@app.cell
def _(env, gamma, np):
    def train_mlp_dqn(env, gamma, hidden=32, episodes=300, max_steps=60,
                      buffer=3000, batch=32, lr=0.03, sync=100, seed=1,
                      n_snaps=30):
        rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        W1 = rng.standard_normal((nS, hidden)) * np.sqrt(2.0 / nS)
        b1 = np.zeros(hidden)
        W2 = rng.standard_normal((hidden, nA)) * np.sqrt(2.0 / hidden)
        b2 = np.zeros(nA)
        tW1, tb1, tW2, tb2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
        I = np.eye(nS)

        def forward(X, p1, q1, p2, q2):
            _h_pre = X @ p1 + q1
            _h = np.maximum(_h_pre, 0.0)
            return _h_pre, _h, _h @ p2 + q2

        buf, returns, snaps = [], [], []
        snap_at = set(int(x) for x in np.linspace(0, episodes, n_snaps,
                                                  endpoint=False))
        step_count = 0
        for ep in range(episodes):
            if ep in snap_at:
                _, _, _Qall = forward(I, W1, b1, W2, b2)
                snaps.append((ep, _Qall.copy()))
            eps = max(0.05, 1.0 - ep / (episodes * 0.6))
            s = env.reset(int(rng.integers(nS)))
            G = 0.0
            for t in range(max_steps):
                if rng.random() < eps:
                    a = int(rng.integers(nA))
                else:
                    _, _, _qs = forward(I[s], W1, b1, W2, b2)
                    a = int(_qs.argmax())
                ns, r, done, _ = env.step(a)
                buf.append((s, a, r, ns, float(done)))
                if len(buf) > buffer:
                    buf.pop(0)
                s = ns
                G += r
                step_count += 1
                if len(buf) >= batch:
                    _idx = rng.integers(0, len(buf), size=batch)
                    _bs = np.array([buf[i][0] for i in _idx])
                    _ba = np.array([buf[i][1] for i in _idx])
                    _br = np.array([buf[i][2] for i in _idx], dtype=float)
                    _bns = np.array([buf[i][3] for i in _idx])
                    _bd = np.array([buf[i][4] for i in _idx])
                    X = I[_bs]
                    _, _, _qtns = forward(I[_bns], tW1, tb1, tW2, tb2)
                    _target = _br + gamma * (1 - _bd) * _qtns.max(1)
                    _h_pre, _h, _q = forward(X, W1, b1, W2, b2)
                    _err = _q[np.arange(batch), _ba] - _target
                    _dq = np.zeros_like(_q)
                    _dq[np.arange(batch), _ba] = 2.0 * _err / batch
                    _dW2 = _h.T @ _dq
                    _db2 = _dq.sum(0)
                    _dh_pre = (_dq @ W2.T) * (_h_pre > 0)
                    _dW1 = X.T @ _dh_pre
                    _db1 = _dh_pre.sum(0)
                    for _g in (_dW1, _db1, _dW2, _db2):
                        np.clip(_g, -5.0, 5.0, out=_g)
                    W1 -= lr * _dW1; b1 -= lr * _db1
                    W2 -= lr * _dW2; b2 -= lr * _db2
                if step_count % sync == 0:
                    tW1, tb1, tW2, tb2 = W1.copy(), b1.copy(), W2.copy(), b2.copy()
                if done:
                    break
            returns.append(G)
        _, _, _Qfin = forward(I, W1, b1, W2, b2)
        snaps.append((episodes, _Qfin.copy()))
        return snaps, returns

    return (train_mlp_dqn,)


@app.cell
def _(env, gamma, train_mlp_dqn):
    dqn_snaps, dqn_returns = train_mlp_dqn(env, gamma, episodes=300, seed=1)
    return dqn_returns, dqn_snaps


@app.cell
def _(dqn_snaps, mo):
    dqn_slider = mo.ui.slider(0, len(dqn_snaps) - 1, step=1, value=0,
                              label="训练快照（越往右训练越久）", show_value=True,
                              full_width=True)
    dqn_slider
    return (dqn_slider,)


@app.cell
def _(
    dqn_returns,
    dqn_slider,
    dqn_snaps,
    env,
    greedy_policy_from_q,
    mo,
    np,
    pi_ep,
    plt,
):
    _idx = min(dqn_slider.value, len(dqn_snaps) - 1)
    _ep_at, _Q = dqn_snaps[_idx]
    _pi = greedy_policy_from_q(_Q)
    _agree = (_pi.argmax(1) == pi_ep.argmax(1)).mean()
    _grid = env.plot_policy(_pi, _Q.max(1),
                            title=f"MLP-DQN @ {_ep_at} eps (与 π* 一致 {_agree:.0%})",
                            figsize=(4.4, 4.4))
    _fig, _ax = plt.subplots(figsize=(5.0, 3.2))
    _rr = np.convolve(dqn_returns, np.ones(20) / 20, mode="valid")
    _ax.plot(_rr, color="#1976d2", lw=1.2)
    _ax.set_xlabel("episode"); _ax.set_ylabel("每回合回报 (滑动平均)")
    _ax.set_title("MLP-DQN 学习曲线"); _ax.grid(True, alpha=0.3)
    _fig.tight_layout()
    mo.hstack([_grid, _fig], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结
    - 函数近似把"查表"变成"拟合"：用 $\phi(s)^\top w$ 表示值，参数远少于状态数。
    - **特征族决定上限**（例子 1/2）：阶数/类型不够，TD 再训练也突破不了最小二乘下界。
    - 线性近似可做控制（例子 3），但比表格更不稳定、且不逐格精确；
      换成神经网络即得 **DQN**，靠**经验回放 + 目标网络**稳定训练（例子 4）。
    - 以上仍是"**学值函数，再贪心导出策略**"。下一章直接对**策略**做梯度上升。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
