import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 6 章 · 书中例子复现 (Worked Examples)

    复现赵世钰《强化学习的数学原理》**第 6 章**的插图例子，全部做成可交互的。
    本章是**随机近似 (Stochastic Approximation)** 的基础——为下一章的 TD 方法做铺垫。

    | 例子 | 书中图/节 | 演示什么 |
    |---|---|---|
    | 1. 均值估计（增量式） | §6.1 | $w_{k+1}=w_k-\frac1k(w_k-x_k)$ 就是在线求平均 |
    | 2. RM 算法 | Fig 6.3 | 只用含噪观测 $\tilde g=g+\eta$ 求 $g(w)=w^3-5=0$ 的根 |
    | 3. RM 收敛直觉 | Fig 6.4 | $g(w)=\tanh(w-1)$，看 $w_k$ 沿曲线逼近根 |
    | 4. SGD vs 小批量 GD | Fig 6.5 | 平面均值估计，m=1/5/50 收敛速度对比 |
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · 均值估计的增量式算法 (§6.1)
    要估计 $\mathbb{E}[X]$，最直接是收集全部样本再平均 $\bar x=\frac1n\sum x_i$。
    但**增量式**写法只需来一个样本更新一次：
    $$w_{k+1}=w_k-\tfrac1k(w_k-x_k).$$
    可以验证它恰好等于前 $k$ 个样本的平均 $\frac1k\sum_{i=1}^k x_i$，按大数定律收敛到 $\mathbb{E}[X]$。
    把系数换成一般的 $\alpha_k$ 也仍收敛（这正是后面 TD 的雏形）。
    """)
    return


@app.cell
def _(mo):
    n_slider = mo.ui.slider(10, 2000, step=10, value=500,
                            label="样本数 n", show_value=True, full_width=True)
    seed1_slider = mo.ui.slider(0, 30, step=1, value=0,
                                label="随机种子", show_value=True, full_width=True)
    alpha_dropdown = mo.ui.dropdown(
        options={"αk = 1/k（= 真·平均）": "1/k", "αk = 0.1（常数步长）": "const"},
        value="αk = 1/k（= 真·平均）", label="步长 αk")
    mo.vstack([n_slider, seed1_slider, alpha_dropdown])
    return alpha_dropdown, n_slider, seed1_slider


@app.cell
def _(alpha_dropdown, mo, n_slider, np, plt, seed1_slider):
    _rng = np.random.default_rng(seed1_slider.value)
    _n = n_slider.value
    # X 取 {0..9} 均匀，真均值 E[X]=4.5
    _xs = _rng.integers(0, 10, size=_n).astype(float)
    _mean = 4.5
    _w = 0.0
    _hist = []
    for _k in range(1, _n + 1):
        _a = 1.0 / _k if alpha_dropdown.value == "1/k" else 0.1
        _w = _w - _a * (_w - _xs[_k - 1])
        _hist.append(_w)
    _run_avg = np.cumsum(_xs) / np.arange(1, _n + 1)
    _fig, _ax = plt.subplots(figsize=(5.6, 3.2))
    _ax.plot(_hist, color="#e53935", lw=1.2, label="增量式 wk")
    _ax.plot(_run_avg, color="#1976d2", lw=1.0, ls="--", label="直接平均")
    _ax.axhline(_mean, color="gray", lw=0.8, label="E[X]=4.5")
    _ax.set_xlabel("样本/迭代 k"); _ax.set_ylabel("估计值")
    _ax.set_title("均值估计：增量式 vs 直接平均")
    _ax.legend(); _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""**n={_n} 时**：增量式估计 wk = **{_hist[-1]:.3f}**，
真均值 E[X] = 4.5。

- 选 **αk=1/k**：增量式与"直接平均"两条线完全重合——增量式就是在线平均。
- 选 **αk=0.1 常数**：估计会一直"抖动"、不会完全收敛，但能跟踪——
  这正是非平稳问题里常用常数步长的原因。""")
    ], widths=[3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 2 · Robbins-Monro 算法 (Fig 6.3)
    要解 $g(w)=0$，但**只能拿到含噪观测** $\tilde g(w,\eta)=g(w)+\eta$。
    RM 算法：$w_{k+1}=w_k-a_k\,\tilde g(w_k,\eta_k)$，**不需要 $g$ 的表达式或导数**。

    书中例子：$g(w)=w^3-5$，真根 $5^{1/3}\approx1.71$；$\eta\sim\mathcal N(0,1)$，
    $w_1=0,\ a_k=1/k$。即使观测带噪，$w_k$ 仍收敛到真根。**调噪声强度和种子**试试。
    """)
    return


@app.cell
def _(mo):
    noise_slider = mo.ui.slider(0.0, 3.0, step=0.25, value=1.0,
                                label="观测噪声标准差 σ", show_value=True,
                                full_width=True)
    seed2_slider = mo.ui.slider(0, 30, step=1, value=0,
                                label="随机种子", show_value=True, full_width=True)
    mo.vstack([noise_slider, seed2_slider])
    return noise_slider, seed2_slider


@app.cell
def _(mo, noise_slider, np, plt, seed2_slider):
    _rng = np.random.default_rng(seed2_slider.value)
    _root = 5 ** (1 / 3)
    _w = 0.0
    _K = 50
    _ws, _etas = [], []
    for _k in range(1, _K + 1):
        _eta = _rng.normal(0, noise_slider.value)
        _g = _w**3 - 5 + _eta          # 含噪观测
        _a = 1.0 / (_k + 9)            # 三次方增长很猛，用更温和的步长保证稳定
        _w = _w - _a * _g
        _w = float(np.clip(_w, -50, 50))
        _ws.append(_w); _etas.append(_eta)
    _fig, _axs = plt.subplots(2, 1, figsize=(5.6, 4.0), sharex=True)
    _axs[0].plot(range(1, _K + 1), _ws, "-o", ms=3, color="#e53935")
    _axs[0].axhline(_root, color="gray", ls="--", lw=0.8)
    _axs[0].set_ylabel("估计根 wk")
    _axs[0].set_title(f"RM 求解 w³−5=0，真根≈{_root:.3f}")
    _axs[1].plot(range(1, _K + 1), _etas, "-", color="#888", lw=0.8)
    _axs[1].axhline(0, color="gray", lw=0.6)
    _axs[1].set_ylabel("观测噪声 ηk"); _axs[1].set_xlabel("迭代 k")
    _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""**50 步后** wk = **{_ws[-1]:.3f}**，真根 5^(1/3) ≈ {_root:.3f}。

RM 算法的精妙之处：
- 完全**不知道** g 的表达式，只用输入 w 和带噪输出；
- 步长 $a_k=1/(k+9)$ 满足 $\\sum a_k=\\infty,\\ \\sum a_k^2<\\infty$，
  使噪声被逐渐平均掉而仍能前进；这里 g(w)=w³−5 增长很快，
  故取比 1/k 略温和的步长以避免发散。

噪声 σ 越大，前期抖动越凶，但仍会收敛。""")
    ], widths=[3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · RM 为什么收敛 (Fig 6.4)
    取 $g(w)=\tanh(w-1)$，真根 $w^*=1$，$w_1=3$，$a_k=1/k$，并令噪声为 0。
    直觉：
    - 当 $w_k>w^*$ 时 $g(w_k)>0$，更新 $w_{k+1}=w_k-a_kg(w_k)<w_k$，更靠近 $w^*$；
    - 当 $w_k<w^*$ 时 $g(w_k)<0$，更新使 $w_{k+1}>w_k$，也更靠近 $w^*$。

    下图把每个 $w_k$ 画在曲线 $g(w)$ 上，**拖动迭代数**看它一步步走向根。
    """)
    return


@app.cell
def _(mo):
    step3_slider = mo.ui.slider(1, 20, step=1, value=1,
                                label="显示前几步 wk", show_value=True,
                                full_width=True)
    w1_slider = mo.ui.slider(-2.0, 4.0, step=0.5, value=3.0,
                             label="初值 w1", show_value=True, full_width=True)
    mo.vstack([w1_slider, step3_slider])
    return step3_slider, w1_slider


@app.cell
def _(mo, np, plt, step3_slider, w1_slider):
    _w = w1_slider.value
    _ws = [_w]
    for _k in range(1, 21):
        _g = np.tanh(_w - 1.0)
        _w = _w - (1.0 / _k) * _g
        _ws.append(_w)
    _n = step3_slider.value
    _grid = np.linspace(-2.5, 4.5, 200)
    _fig, _ax = plt.subplots(figsize=(5.6, 3.4))
    _ax.plot(_grid, np.tanh(_grid - 1.0), color="#1976d2", label="g(w)=tanh(w−1)")
    _ax.axhline(0, color="gray", lw=0.7)
    _ax.axvline(1.0, color="gray", ls="--", lw=0.7, label="真根 w*=1")
    _wx = np.array(_ws[:_n + 1])
    _ax.plot(_wx, np.tanh(_wx - 1.0), "o-", color="#e53935", ms=5,
             label="wk 轨迹")
    for _i, _wv in enumerate(_wx):
        _ax.annotate(f"w{_i+1}", (_wv, np.tanh(_wv - 1.0)),
                     textcoords="offset points", xytext=(0, 7), fontsize=7)
    _ax.set_xlabel("w"); _ax.set_ylabel("g(w)")
    _ax.set_title("RM 收敛直觉：wk 沿 g(w) 逼近根")
    _ax.legend(fontsize=8); _fig.tight_layout()
    mo.hstack([
        _fig,
        mo.md(f"""**前 {_n} 步**：w{_n+1} = **{_ws[_n]:.4f}**（真根 = 1）。

不论初值在根的左边还是右边，$g$ 的单调性保证每一步都把 $w_k$ 往根的方向推。
试着把初值 w1 拖到根的左侧（如 −2），看轨迹从下方爬上来。""")
    ], widths=[3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 4 · SGD vs 小批量 GD (Fig 6.5)
    平面均值估计：$X\in\mathbb R^2$ 在边长 20、中心原点的正方形里均匀分布，$\mathbb E[X]=0$。
    用 100 个 i.i.d. 样本，最小化 $J(w)=\frac1{2n}\sum\|w-x_i\|^2$（最优解 = 样本均值）。

    三种梯度下降，$\alpha_k=1/k$：
    - **SGD (m=1)**：每步只用 1 个样本，随机性大、收敛慢；
    - **MBGD (m=5 / m=50)**：每步用一小批，随机性被平均掉，收敛更快。

    左图是估计点在平面上的轨迹，右图是到真值的距离。**换种子**看随机性。
    """)
    return


@app.cell
def _(mo):
    seed4_slider = mo.ui.slider(0, 30, step=1, value=0,
                                label="随机种子", show_value=True, full_width=True)
    seed4_slider
    return (seed4_slider,)


@app.cell
def _(mo, np, plt, seed4_slider):
    _rng = np.random.default_rng(seed4_slider.value)
    _samples = _rng.uniform(-10, 10, size=(100, 2))   # 边长 20 的正方形
    _mean = _samples.mean(0)
    _w0 = np.array([18.0, -15.0])                      # 远离真值的初值

    def _mbgd(m, steps=40):
        _w = _w0.copy()
        _traj = [_w.copy()]
        for _k in range(1, steps + 1):
            _idx = _rng.integers(0, 100, size=m)
            _grad = (_w - _samples[_idx]).mean(0)      # ∇ = w − batch_mean
            _w = _w - (1.0 / _k) * _grad
            _traj.append(_w.copy())
        return np.array(_traj)

    _t1 = _mbgd(1)
    _t5 = _mbgd(5)
    _t50 = _mbgd(50)
    _fig, _axs = plt.subplots(1, 2, figsize=(8.4, 3.6))
    _axs[0].scatter(_samples[:, 0], _samples[:, 1], s=8, color="#ccc", label="样本")
    _axs[0].plot(_t1[:, 0], _t1[:, 1], "-o", ms=2, color="#e53935", label="SGD m=1")
    _axs[0].plot(_t5[:, 0], _t5[:, 1], "-o", ms=2, color="#ff9800", label="MBGD m=5")
    _axs[0].plot(_t50[:, 0], _t50[:, 1], "-o", ms=2, color="#1976d2", label="MBGD m=50")
    _axs[0].plot(0, 0, "k*", ms=14, label="E[X]=0")
    _axs[0].legend(fontsize=7); _axs[0].set_title("平面上的估计轨迹")
    _axs[0].set_aspect("equal")
    for _t, _c, _l in [(_t1, "#e53935", "m=1"), (_t5, "#ff9800", "m=5"),
                       (_t50, "#1976d2", "m=50")]:
        _axs[1].plot(np.linalg.norm(_t, axis=1), color=_c, label=_l)
    _axs[1].set_xlabel("迭代步"); _axs[1].set_ylabel("到真值的距离")
    _axs[1].set_title("收敛速度"); _axs[1].legend(fontsize=8)
    _fig.tight_layout()
    mo.vstack([
        _fig,
        mo.md(f"""m 越大，每步梯度越接近真实梯度、轨迹越平滑、收敛越快
（m=50 最快，m=1 最慢但前期下降也很快）。样本均值 = ({_mean[0]:.2f}, {_mean[1]:.2f})，
接近真值 (0,0)。""")
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结
    - **增量式均值估计** $w_{k+1}=w_k-\alpha_k(w_k-x_k)$ 是随机近似的最简形式（§6.1）。
    - **RM 算法**只靠含噪观测就能求根，无需函数表达式；步长需满足
      $\sum a_k=\infty,\ \sum a_k^2<\infty$（Fig 6.3/6.4）。
    - **SGD 是 RM 的特例**；小批量 GD 介于 SGD 与全量 GD 之间，batch 越大越稳越快（Fig 6.5）。
    下一章的 **TD 学习**正是把这些随机近似思想用到状态值/动作值的在线估计上。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
