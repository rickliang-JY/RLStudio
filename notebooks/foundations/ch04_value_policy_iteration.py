import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 第 4 章 · 值迭代与策略迭代 (Value / Policy Iteration)

        本章给出三个**已知模型**时求最优策略的算法，都是第 3 章 BOE 的落地：

        | 算法 | 每轮做什么 |
        |---|---|
        | **值迭代 VI** | 直接迭代 $v\leftarrow\max_a q$，最后提取策略 |
        | **策略迭代 PI** | 反复 (精确评估当前策略 → 贪心改进) |
        | **截断策略迭代** | 评估只迭代 $m$ 次（不解到底）|

        下面每个算法都配**逐步回放滑块**：拖动即可看到 $v$ 和策略一轮轮地变化。
        """
    )
    return


@app.cell
def _():
    import numpy as np
    from gridworld import classic_example, greedy_policy_from_q, q_from_v
    env = classic_example()
    return classic_example, env, greedy_policy_from_q, np, q_from_v


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🎛️ 全局参数：γ 与**禁区惩罚**（三个算法联动）
        - $\gamma$：折扣因子（本网格里最优策略对它几乎不敏感，主要影响数值大小）。
        - **禁区惩罚 $r_{\text{forbidden}}$**：真正能让**策略箭头改道**的旋钮——
          惩罚轻则抄近路穿禁区，惩罚重则绕道避开。拖动它看 VI / PI 的最终策略整体改变。
        """
    )
    return


@app.cell
def _(mo):
    gamma_slider = mo.ui.slider(
        start=0.1, stop=0.99, step=0.01, value=0.9,
        label="折扣因子 γ", show_value=True, full_width=True,
    )
    rforbid_slider = mo.ui.slider(
        start=-10.0, stop=-0.2, step=0.2, value=-1.0,
        label="禁区惩罚 r_forbidden（越负越不敢穿越禁区）",
        show_value=True, full_width=True,
    )
    mo.vstack([gamma_slider, rforbid_slider])
    return gamma_slider, rforbid_slider


@app.cell
def _(classic_example, rforbid_slider):
    env_m = classic_example()
    env_m.r_forbidden = rforbid_slider.value     # 同一布局，只改禁区惩罚
    P, R = env_m.build_model()
    return P, R


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 算法 1：值迭代 (Value Iteration) — 逐步回放
        每一步 $v(s)\leftarrow\max_a[r(s,a)+\gamma\sum p\,v(s')]$。
        拖动"迭代步"看格子数值被一步步抬高、箭头逐渐稳定。
        """
    )
    return


@app.cell
def _(P, R, greedy_policy_from_q, np):
    def vi_history(P, R, gamma, tol=1e-9, max_iter=300):
        v = np.zeros(R.shape[0])
        hist = []
        for _ in range(max_iter):
            q = R + gamma * P.dot(v)
            hist.append((v.copy(), greedy_policy_from_q(q)))
            vn = q.max(1)
            if np.max(np.abs(vn - v)) < tol:
                v = vn
                break
            v = vn
        hist.append((v.copy(), greedy_policy_from_q(R + gamma * P.dot(v))))
        return hist
    return (vi_history,)


@app.cell
def _(P, R, gamma_slider, vi_history):
    vi_hist = vi_history(P, R, gamma_slider.value)
    vi_steps = len(vi_hist) - 1
    return vi_hist, vi_steps


@app.cell
def _(mo, vi_steps):
    vi_slider = mo.ui.slider(0, vi_steps, step=1, value=0,
                             label="值迭代 · 迭代步 k", show_value=True, full_width=True)
    vi_slider
    return (vi_slider,)


@app.cell
def _(env, gamma_slider, np, rforbid_slider, vi_hist, vi_slider, vi_steps):
    k = min(vi_slider.value, len(vi_hist) - 1)
    vk, pik = vi_hist[k]
    sub = "v_0 = 0" if k == 0 else f"较上一步 Δ={np.max(np.abs(vk - vi_hist[k-1][0])):.3f}"
    env.plot_policy(pik, vk,
                    title=(f"值迭代 第 {k}/{vi_steps} 步 (γ={gamma_slider.value:.2f}, "
                           f"禁区={rforbid_slider.value:.1f})\n{sub}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 算法 2：策略迭代 (Policy Iteration) — 逐步回放
        每**一轮** = 完整评估当前策略得到 $v_\pi$ → 对 $v_\pi$ 贪心得到新策略。
        通常只需很少几轮就收敛。拖动看每轮策略如何整体跳变。
        """
    )
    return


@app.cell
def _(P, R, greedy_policy_from_q, np, q_from_v):
    def pi_history(P, R, gamma, tol=1e-9):
        nS, nA = R.shape
        pi = np.ones((nS, nA)) / nA
        v = np.zeros(nS)
        hist = []
        while True:
            # 精确策略评估
            P_pi = np.einsum("sa,sax->sx", pi, P)
            r_pi = np.einsum("sa,sa->s", pi, R)
            v = np.linalg.solve(np.eye(nS) - gamma * P_pi, r_pi)
            hist.append((v.copy(), pi.copy()))
            pi_new = greedy_policy_from_q(q_from_v(v, P, R, gamma))
            if np.array_equal(pi_new, pi):
                break
            pi = pi_new
        return hist
    return (pi_history,)


@app.cell
def _(P, R, gamma_slider, pi_history):
    pi_hist = pi_history(P, R, gamma_slider.value)
    pi_rounds = len(pi_hist) - 1
    return pi_hist, pi_rounds


@app.cell
def _(mo, pi_rounds):
    pi_slider = mo.ui.slider(0, pi_rounds, step=1, value=0,
                             label="策略迭代 · 轮次", show_value=True, full_width=True)
    pi_slider
    return (pi_slider,)


@app.cell
def _(env, gamma_slider, pi_hist, pi_rounds, pi_slider, rforbid_slider):
    r = min(pi_slider.value, len(pi_hist) - 1)
    vr, pir = pi_hist[r]
    env.plot_policy(pir, vr,
                    title=(f"策略迭代 第 {r}/{pi_rounds} 轮 (γ={gamma_slider.value:.2f}, "
                           f"禁区={rforbid_slider.value:.1f})"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        对比一下：值迭代往往要十几步，策略迭代只要**两三轮**就收敛——
        因为它每轮把策略评估"解到底"。代价是每轮更贵。

        ## 算法 3：截断策略迭代 — 用滑块调"评估步数 m"
        VI 是"评估只走 1 步"，PI 是"评估走到收敛"，截断 PI 取中间：评估只迭代 $m$ 步。
        拖动 $m$，观察相同总计算量下到 $v^*$ 的收敛速度。
        """
    )
    return


@app.cell
def _(mo):
    m_slider = mo.ui.slider(1, 50, step=1, value=5,
                            label="截断评估步数 m", show_value=True, full_width=True)
    m_slider
    return (m_slider,)


@app.cell
def _(P, R, gamma_slider, greedy_policy_from_q, m_slider, np, q_from_v, vi_hist):
    import matplotlib.pyplot as plt
    gamma = gamma_slider.value
    v_star = vi_hist[-1][0]

    def trunc_curve(m, outer=60):
        nS, nA = R.shape
        pi = np.ones((nS, nA)) / nA
        v = np.zeros(nS)
        xs, ys, total = [], [], 0
        for _ in range(outer):
            P_pi = np.einsum("sa,sax->sx", pi, P)
            r_pi = np.einsum("sa,sa->s", pi, R)
            for _ in range(m):
                v = r_pi + gamma * P_pi @ v
                total += 1
            xs.append(total)
            ys.append(np.max(np.abs(v - v_star)))
            pi = greedy_policy_from_q(q_from_v(v, P, R, gamma))
        return xs, ys

    figt, axt = plt.subplots(figsize=(5.8, 3.4))
    for mm, lab in [(1, "m=1 (值迭代)"), (m_slider.value, f"m={m_slider.value} (你选的)"),
                    (100, "m=100 (≈策略迭代)")]:
        xs, ys = trunc_curve(mm)
        axt.semilogy(xs, np.maximum(ys, 1e-12), "o-", ms=3, label=lab)
    axt.set_xlabel("累计 value-update 次数")
    axt.set_ylabel(r"$\|v-v^*\|_\infty$")
    axt.set_title(f"截断策略迭代 (γ={gamma:.2f})")
    axt.legend(fontsize=8)
    axt.grid(True, alpha=0.3)
    figt
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 小结
        三种 model-based 算法殊途同归到 $\pi^*$，区别在"每轮把评估做多准"。
        但它们都需要**已知** $P,R$。下一章起转向 **model-free**：用采样数据来学。
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
