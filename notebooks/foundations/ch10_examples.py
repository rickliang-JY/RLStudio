import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 10 章 · 书中例子复现 (Worked Examples)

    本 notebook 把赵世钰《强化学习的数学原理》第 10 章的**核心例子**逐一复现并做成
    **可交互**的。第 10 章是全书的汇合点 —— **Actor-Critic** = 策略梯度 (actor)
    + 值函数近似 (critic)：actor 出动作，critic 用 TD 在线评估、给 actor
    提供低方差的信号。

    | 例子 | 对应内容 | 演示什么 |
    |---|---|---|
    | 1. QAC | §10.2 | critic 学 $q$，actor 用 $q$ 当权重；最简单的 AC |
    | 2. A2C | §10.3 | 用 TD error $\delta$ 当优势，方差更低 |
    | 3. 重要性采样 | §10.4 / Fig 10.2 | 用"别的分布"的样本估计目标期望 |
    | 4. 离线策略 AC | §10.4 | 行为策略采数据，重要性权重纠偏，照样学到 $\pi^\*$ |

    **Actor-Critic 每步 $(s,a,r,s')$：**
    - Critic：$\hat v(s)\leftarrow\hat v(s)+\alpha_w\,\delta$，其中
      $\delta=r+\gamma\hat v(s')-\hat v(s)$。
    - Actor：$\theta_s\leftarrow\theta_s+\alpha_\theta\,\delta\,(e_a-\pi(\cdot|s))$。

    > 第 1、2、4 例的 5×5 网格（episodic）奖励：
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
    # episodic 最优策略——作为各 AC 算法的对照
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
    ## 例子 1 · QAC (Q Actor-Critic, §10.2) — 🎛️ 逐步回放

    最简单的 actor-critic。每一步 $(s,a,r,s',a')$：
    - **Critic** 用 Sarsa 更新 $q$：$q(s,a)\leftarrow q(s,a)+\alpha_w[r+\gamma q(s',a')-q(s,a)]$。
    - **Actor** 用策略梯度更新，权重就是 critic 的 $q(s,a)$：
      $\theta_s\leftarrow\theta_s+\alpha_\theta\,q(s,a)\,(e_a-\pi(\cdot|s))$。

    拖动滑块看 actor 的策略与 critic 的值（格子上方数字）一起收敛。
    """)
    return


@app.cell
def _(env, gamma, np, policy_from_theta, softmax):
    def qac(_env, _gamma, alpha_theta=0.02, alpha_w=0.1,
            episodes=6000, max_steps=80, seed=1, n_snaps=40):
        _rng = np.random.default_rng(seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _theta = np.zeros((_nS, _nA))
        _Q = np.zeros((_nS, _nA))
        _returns = []
        _snaps = []
        _snap_at = set(int(x) for x in
                       np.linspace(0, episodes, n_snaps, endpoint=False))
        for _ep in range(episodes):
            if _ep in _snap_at:
                _snaps.append((_ep, policy_from_theta(_theta), _Q.max(1).copy()))
            _s = _env.reset(int(_rng.integers(_nS)))
            _a = _rng.choice(_nA, p=softmax(_theta[_s]))
            _ret, _disc = 0.0, 1.0
            for _t in range(max_steps):
                _ns, _r, _done, _ = _env.step(_a)
                _na = _rng.choice(_nA, p=softmax(_theta[_ns]))
                _target = _r + (0.0 if _done else _gamma * _Q[_ns, _na])
                _Q[_s, _a] += alpha_w * (_target - _Q[_s, _a])
                _p = softmax(_theta[_s])
                _gl = -_p
                _gl[_a] += 1.0
                _theta[_s] += alpha_theta * _disc * _Q[_s, _a] * _gl
                _ret += _r
                _disc *= _gamma
                _s, _a = _ns, _na
                if _done:
                    break
            _returns.append(_ret)
        _snaps.append((episodes, policy_from_theta(_theta), _Q.max(1).copy()))
        return _returns, _snaps

    ret_qac, qac_snaps = qac(env, gamma, episodes=6000, seed=1, n_snaps=40)
    return qac_snaps, ret_qac


@app.cell
def _(mo, qac_snaps):
    qac_replay = mo.ui.slider(0, len(qac_snaps) - 1, step=1, value=0,
                              label="QAC 训练快照（越往右训练越久）", show_value=True,
                              full_width=True)
    qac_replay
    return (qac_replay,)


@app.cell
def _(env, pi_opt, qac_replay, qac_snaps):
    _idx = min(qac_replay.value, len(qac_snaps) - 1)
    _ep_at, _pi_qac, _vq = qac_snaps[_idx]
    _agree = (_pi_qac.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(_pi_qac, _vq,
                    title=f"QAC @ {_ep_at} episodes（与 π* 一致率 {_agree:.0%}）")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · A2C (Advantage Actor-Critic, §10.3) — 🎛️ 逐步回放

    Critic 只学**状态值** $\hat v(s)$，用 **TD error** 当作优势的无偏估计：
    $$ \delta_t=r+\gamma\hat v(s')-\hat v(s). $$
    $\delta$ **同时**充当 critic 的学习信号和 actor 的优势信号 —— 这是 A2C
    的精妙之处。相比 QAC 学整张 $q$ 表，A2C 只学 $\hat v$，参数更少、方差更低。
    """)
    return


@app.cell
def _(env, gamma, np, policy_from_theta, softmax):
    def a2c(_env, _gamma, alpha_theta=0.05, alpha_w=0.2,
            episodes=6000, max_steps=80, seed=1, n_snaps=40):
        _rng = np.random.default_rng(seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _theta = np.zeros((_nS, _nA))
        _V = np.zeros(_nS)
        _returns = []
        _snaps = []
        _snap_at = set(int(x) for x in
                       np.linspace(0, episodes, n_snaps, endpoint=False))
        for _ep in range(episodes):
            if _ep in _snap_at:
                _snaps.append((_ep, policy_from_theta(_theta), _V.copy()))
            _s = _env.reset(int(_rng.integers(_nS)))
            _ret, _disc = 0.0, 1.0
            for _t in range(max_steps):
                _p = softmax(_theta[_s])
                _a = _rng.choice(_nA, p=_p)
                _ns, _r, _done, _ = _env.step(_a)
                _delta = _r + (0.0 if _done else _gamma * _V[_ns]) - _V[_s]
                _V[_s] += alpha_w * _delta
                _gl = -_p
                _gl[_a] += 1.0
                _theta[_s] += alpha_theta * _disc * _delta * _gl
                _ret += _r
                _disc *= _gamma
                _s = _ns
                if _done:
                    break
            _returns.append(_ret)
        _snaps.append((episodes, policy_from_theta(_theta), _V.copy()))
        return _returns, _snaps

    ret_a2c, a2c_snaps = a2c(env, gamma, episodes=6000, seed=1, n_snaps=40)
    return a2c_snaps, ret_a2c


@app.cell
def _(a2c_snaps, mo):
    a2c_replay = mo.ui.slider(0, len(a2c_snaps) - 1, step=1, value=0,
                              label="A2C 训练快照（越往右训练越久）", show_value=True,
                              full_width=True)
    a2c_replay
    return (a2c_replay,)


@app.cell
def _(a2c_replay, a2c_snaps, env, pi_opt):
    _idx = min(a2c_replay.value, len(a2c_snaps) - 1)
    _ep_at, _pi_a2c, _V = a2c_snaps[_idx]
    _agree = (_pi_a2c.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(_pi_a2c, _V,
                    title=f"A2C @ {_ep_at} episodes（与 π* 一致率 {_agree:.0%}）")
    return


@app.cell
def _(np, plt, ret_a2c, ret_qac):
    def _smooth(x, k=50):
        return np.convolve(np.asarray(x, float), np.ones(k) / k, mode="valid")

    _figC, _axC = plt.subplots(figsize=(6, 3.2))
    _axC.plot(_smooth(ret_qac), label="QAC")
    _axC.plot(_smooth(ret_a2c), label="A2C (advantage)")
    _axC.set_xlabel("episode")
    _axC.set_ylabel("每回合总奖励 (滑动平均)")
    _axC.set_title("Actor-Critic 学习曲线对比")
    _axC.legend()
    _axC.grid(True, alpha=0.3)
    _figC
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · 重要性采样 (§10.4, Fig 10.2)

    离线策略学习的数学基石。设随机变量 $X\in\{+1,-1\}$。我们想估计
    **目标分布** $p_0$ 下的期望 $\mathbb{E}_{p_0}[X]$，但手里的样本却来自
    **另一个行为分布** $p_1$。直接平均会收敛到 $\mathbb{E}_{p_1}[X]$（错的）。

    **重要性采样**给每个样本乘权重 $w(x)=\dfrac{p_0(x)}{p_1(x)}$，则
    $$ \mathbb{E}_{p_1}\!\big[w(X)\,X\big]
       =\sum_x p_1(x)\frac{p_0(x)}{p_1(x)}x=\mathbb{E}_{p_0}[X]. $$
    下面固定目标 $p_0(X\!=\!+1)=0.5$（即 $\mathbb{E}_{p_0}[X]=0$），拖动**行为**
    分布 $p_1(X\!=\!+1)$，看朴素平均跑偏、而 IS 估计始终收敛到 0。
    """)
    return


@app.cell
def _(mo):
    behav_p = mo.ui.slider(0.55, 0.95, step=0.05, value=0.8,
                           label="行为分布 p₁(X=+1)", show_value=True,
                           full_width=True)
    behav_p
    return (behav_p,)


@app.cell
def _(behav_p, np, plt):
    _p0 = 0.5            # 目标分布：E_p0[X] = 0
    _p1 = behav_p.value  # 行为分布
    _rng = np.random.default_rng(0)
    _N = 2000
    _xs = np.where(_rng.random(_N) < _p1, 1.0, -1.0)
    _w = np.where(_xs > 0, _p0 / _p1, (1 - _p0) / (1 - _p1))
    _k = np.arange(1, _N + 1)
    _naive = np.cumsum(_xs) / _k
    _is_est = np.cumsum(_w * _xs) / _k

    _fig3, _ax = plt.subplots(figsize=(6.4, 3.4))
    _ax.plot(_naive, color="#5fa85f", label="朴素平均 → 偏向行为均值")
    _ax.plot(_is_est, color="#d8754f", label="重要性采样 → 目标均值")
    _ax.axhline(0.0, color="black", ls="--", lw=1, label="目标 E_p0[X]=0")
    _ax.axhline(2 * _p1 - 1, color="gray", ls=":", lw=1,
                label=f"行为均值 E_p1[X]={2*_p1-1:.1f}")
    _ax.set_xlabel("样本数")
    _ax.set_ylabel("估计值")
    _ax.set_title(f"重要性采样（行为 p1={_p1:.2f}）")
    _ax.legend(fontsize=8, loc="upper right")
    _ax.grid(True, alpha=0.3)
    _fig3.tight_layout()
    _fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 4 · 离线策略 Actor-Critic (§10.4) — 🎛️ 重要性权重开关

    把例子 3 的重要性采样用进 actor-critic。数据全部由一个**固定的行为策略**
    $\beta$（这里取**均匀随机**）产生，目标策略 $\pi$ 却能学好。每步乘上
    **重要性比率** $\rho=\dfrac{\pi(a|s)}{\beta(a|s)}$ 纠偏：
    $$ \hat v(s)\mathrel{+}=\alpha_w\,\rho\,\delta,\qquad
       \theta_s\mathrel{+}=\alpha_\theta\,\rho\,\delta\,(e_a-\pi(\cdot|s)). $$
    勾选/取消重要性权重，对比有无纠偏时学到的策略。
    """)
    return


@app.cell
def _(mo):
    use_is = mo.ui.checkbox(value=True, label="启用重要性权重 ρ=π/β（纠偏）")
    offpol_run = mo.ui.run_button(label="▶ 训练离线策略 A2C")
    mo.vstack([use_is, offpol_run])
    return offpol_run, use_is


@app.cell
def _(env, gamma, np, policy_from_theta, softmax):
    def offpolicy_a2c(_env, _gamma, alpha_theta=0.05, alpha_w=0.2,
                      episodes=6000, max_steps=80, seed=1, use_is=True):
        _rng = np.random.default_rng(seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _theta = np.zeros((_nS, _nA))
        _V = np.zeros(_nS)
        _beta = np.ones(_nA) / _nA            # 均匀随机行为策略
        for _ep in range(episodes):
            _s = _env.reset(int(_rng.integers(_nS)))
            _disc = 1.0
            for _t in range(max_steps):
                _a = _rng.choice(_nA, p=_beta)
                _ns, _r, _done, _ = _env.step(_a)
                _p = softmax(_theta[_s])
                _rho = (_p[_a] / _beta[_a]) if use_is else 1.0
                _delta = _r + (0.0 if _done else _gamma * _V[_ns]) - _V[_s]
                _V[_s] += alpha_w * _rho * _delta
                _gl = -_p
                _gl[_a] += 1.0
                _theta[_s] += alpha_theta * _disc * _rho * _delta * _gl
                _disc *= _gamma
                _s = _ns
                if _done:
                    break
        return policy_from_theta(_theta), _V
    return (offpolicy_a2c,)


@app.cell
def _(env, gamma, mo, offpol_run, offpolicy_a2c, pi_opt, use_is):
    mo.stop(not offpol_run.value,
            mo.md("⬆️ **勾选是否启用重要性权重，点击「▶ 训练」按钮**，这里显示结果。"))
    _pi_off, _V_off = offpolicy_a2c(env, gamma, episodes=6000, seed=1,
                                    use_is=use_is.value)
    _agree = (_pi_off.argmax(1) == pi_opt.argmax(1)).mean()
    _tag = "有" if use_is.value else "无"
    env.plot_policy(
        _pi_off, _V_off,
        title=f"离线策略 A2C（{_tag}重要性纠偏）→ 与 π* 一致率 {_agree:.0%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    数据全来自均匀随机的 $\beta$，但目标策略 $\pi$ 依旧学到接近 $\pi^\*$。
    重要性比率 $\rho$ 让"目标策略更爱选的动作"在更新里占更大权重。

    ## 小结
    - **Actor-Critic** 把"值"（critic）与"策略"（actor）两条主线合一：
      critic 用 TD/随机近似在线评估，actor 用策略梯度直接优化。
    - **QAC** critic 学 $q$；**A2C** critic 只学 $\hat v$、用 TD error $\delta$
      当优势，参数更少、方差更低，是最常用的形式。
    - **重要性采样**是离线策略学习的基石：用行为分布的样本、乘权重
      $p_0/p_1$，无偏地估计目标分布下的期望。
    - 把它用进 AC 就得到**离线策略 actor-critic**；若策略是**确定性**的
      $a=\mu(s;\theta)$，则进一步得到 **DPG / DDPG**（连续动作）。
    - 这正是现代深度强化学习（A2C/A3C、PPO、SAC、DDPG）的统一骨架 —— 全书收官。
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
