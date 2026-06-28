import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 第 6 章 · 随机近似 (Stochastic Approximation)

        本章是连接 MC 与 TD 的**数学桥梁**。它回答一个问题：
        在**只有噪声样本、不知道期望/梯度表达式**的情况下，如何迭代地求解
        方程 $g(w)=0$ 或最小化 $J(w)=\mathbb{E}[f(w,X)]$？

        三个递进的工具：
        1. **增量式均值估计**：求平均值也能写成"逐样本更新"。
        2. **Robbins–Monro (RM)** 算法：求根 $g(w)=0$ 的随机迭代。
        3. **随机梯度下降 (SGD)**：RM 用在梯度上的特例。

        贯穿全章的主角是**步长序列** $\alpha_k$，它必须满足
        $\sum_k\alpha_k=\infty,\ \sum_k\alpha_k^2<\infty$（如 $\alpha_k=1/k$）才能收敛。
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    rng = np.random.default_rng(0)
    return np, plt, rng


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 工具 1：增量式均值估计
        要估计 $\mathbb{E}[X]$，朴素做法是存下所有样本再平均。但其实可以**在线**更新：
        $$ w_{k+1}=w_k+\tfrac1k\,(x_k-w_k). $$
        它等价于前 $k$ 个样本的算术平均，却无需存储历史。把步长 $1/k$ 换成一般 $\alpha_k$
        就得到了所有 RL 增量算法的统一形式 $w_{k+1}=w_k+\alpha_k(\text{目标}-w_k)$。
        """
    )
    return


@app.cell
def _(np, plt, rng):
    true_mean = 3.0
    samples = rng.normal(true_mean, 2.0, size=2000)

    w, est_1k = 0.0, []
    for k, x in enumerate(samples, start=1):
        w = w + (1.0 / k) * (x - w)
        est_1k.append(w)

    fig1, ax1 = plt.subplots(figsize=(5.6, 3.2))
    ax1.plot(est_1k, label=r"增量估计 $w_k$")
    ax1.axhline(true_mean, color="r", ls="--", label="真值 E[X]=3")
    ax1.plot(np.cumsum(samples) / np.arange(1, len(samples) + 1), ":",
             color="gray", label="直接累加平均（应重合）")
    ax1.set_xlabel("样本数 k")
    ax1.set_ylabel("估计值")
    ax1.legend(fontsize=8)
    ax1.set_title("步长 1/k 的增量均值估计")
    fig1
    return (samples,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 工具 2：Robbins–Monro 算法 — 🎛️ 调噪声 / 步长
        要解 $g(w^*)=0$，但只能观测到带噪声的 $\tilde g(w)=g(w)+\eta$。RM 迭代：
        $$ w_{k+1}=w_k-\alpha_k\,\tilde g(w_k). $$
        **例子**：求 $g(w)=\tanh(w-1)=0$ 的根（真根 $w^*=1$）。
        拖动滑块改变**噪声强度 σ**，并切换**步长**为递减 $1/k$ 还是常数——
        看三条不同初值轨迹的收敛/抖动行为如何变化。
        """
    )
    return


@app.cell
def _(mo):
    rm_noise = mo.ui.slider(0.0, 2.0, step=0.05, value=0.3,
                            label="观测噪声强度 σ", show_value=True, full_width=True)
    rm_step = mo.ui.dropdown(options=["递减 1/k", "常数 0.05", "常数 0.2"],
                             value="递减 1/k", label="步长 α_k")
    mo.vstack([rm_noise, rm_step])
    return rm_noise, rm_step


@app.cell
def _(np, plt, rm_noise, rm_step):
    def g(w):
        return np.tanh(w - 1.0)          # 真根在 w=1

    def _alpha(k):
        if rm_step.value == "递减 1/k":
            return 1.0 / k
        return float(rm_step.value.split()[-1])

    def robbins_monro(w0, n=300, noise=0.3, seed=1):
        r = np.random.default_rng(seed)
        w, hist = w0, [w0]
        for k in range(1, n + 1):
            noisy = g(w) + r.normal(0, noise)
            w = w - _alpha(k) * noisy
            hist.append(w)
        return np.array(hist)

    paths = [robbins_monro(w0=w0, noise=rm_noise.value, seed=s)
             for w0, s in [(5, 1), (-3, 2), (8, 3)]]
    fig2, ax2 = plt.subplots(figsize=(5.6, 3.2))
    for p, w0 in zip(paths, [5, -3, 8]):
        ax2.plot(p, label=f"w0={w0}")
    ax2.axhline(1.0, color="r", ls="--", label="真根 w*=1")
    ax2.set_xlabel("迭代 k")
    ax2.set_ylabel(r"$w_k$")
    ax2.legend(fontsize=8)
    ax2.set_title(f"Robbins–Monro (σ={rm_noise.value:.2f}, α={rm_step.value})")
    fig2
    return (g,)


@app.cell
def _(mo):
    mo.md(
        r"""
        三条不同初值的轨迹都收敛到 $w^*=1$。
        **关键洞察**：增量均值估计其实就是 RM 在 $g(w)=w-\mathbb{E}[X]$ 上的特例。
        而下一章的 **TD 算法**，正是 RM 用来求解贝尔曼方程 $v=r_\pi+\gamma P_\pi v$
        （把它写成 $g(v)=0$）的产物。

        ## 工具 3：随机梯度下降 (SGD)
        要最小化 $J(w)=\mathbb{E}[f(w,X)]$。真梯度 $\nabla J=\mathbb{E}[\nabla_w f]$ 算不出，
        就用**单个样本**的梯度代替：
        $$ w_{k+1}=w_k-\alpha_k\,\nabla_w f(w_k,x_k). $$
        以"最小化 $\tfrac12\mathbb{E}[(w-X)^2]$"为例，其最优解恰是 $w^*=\mathbb{E}[X]$，
        梯度为 $w-x$，于是 SGD ⟹ 增量均值估计。三者在此闭环。
        """
    )
    return


@app.cell
def _(mo):
    sgd_alpha = mo.ui.slider(0.005, 0.3, step=0.005, value=0.05,
                             label="常数步长 α", show_value=True, full_width=True)
    sgd_alpha
    return (sgd_alpha,)


@app.cell
def _(np, plt, samples, sgd_alpha):
    def sgd_mean(samples, alpha=None):
        w, hist = 0.0, []
        for k, x in enumerate(samples, start=1):
            step = (1.0 / k) if alpha is None else alpha
            w = w - step * (w - x)        # 梯度 = w - x
            hist.append(w)
        return hist

    fig3, ax3 = plt.subplots(figsize=(5.6, 3.2))
    ax3.plot(sgd_mean(samples), label=r"$\alpha_k=1/k$（收敛）")
    ax3.plot(sgd_mean(samples, alpha=sgd_alpha.value),
             label=fr"$\alpha={sgd_alpha.value:.3f}$ 常数（抖动幅度随 α 变化）")
    ax3.axhline(3.0, color="r", ls="--", label="E[X]=3")
    ax3.set_xlabel("样本数 k")
    ax3.set_ylabel(r"$w_k$")
    ax3.legend(fontsize=8)
    ax3.set_title(f"SGD 估计均值：常数步长 α={sgd_alpha.value:.3f} 的抖动")
    fig3
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        - 递减步长 $1/k$：满足 RM 条件，**精确收敛**。
        - 常数步长 $\alpha=0.05$：不满足 $\sum\alpha_k^2<\infty$，在最优解附近**持续抖动**，
          但好处是能跟踪**非平稳**目标——这正是深度 RL 常用常数步长的原因。

        ## 小结
        | 工具 | 更新式 | 在 RL 中对应 |
        |---|---|---|
        | 增量均值 | $w\leftarrow w+\alpha_k(x_k-w)$ | MC 的在线版 |
        | Robbins–Monro | $w\leftarrow w-\alpha_k\tilde g(w)$ | TD 求解贝尔曼方程 |
        | SGD | $w\leftarrow w-\alpha_k\nabla f(w,x_k)$ | 值函数近似 / 策略梯度 |

        有了这把"随机近似"的钥匙，下一章的 TD 学习就水到渠成。
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
