import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # 第 3 章 · 最优状态值与贝尔曼最优方程 (BOE)

        第 2 章评价**给定**策略；本章寻找**最优**策略 $\pi^*$，
        它的状态值 $v^*(s)=\max_\pi v_\pi(s)$ 对所有 $s$ 同时最大。

        ## 贝尔曼最优方程
        $$ v^*(s)=\max_a\Big(r(s,a)+\gamma\sum_{s'}p(s'|s,a)v^*(s')\Big),\quad
           v^*=\max_\pi (r_\pi+\gamma P_\pi v^*). $$

        右端算子 $f$ 是 $\gamma$-压缩映射，故 BOE 有唯一解，迭代 $v_{k+1}=f(v_k)$ 必收敛
        （这正是**值迭代**）。下面用**可交互**的方式把"每一步迭代里 $v$ 和策略如何变化"看清楚。
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
        ## 🎛️ 交互区 1：拖动滑块，实时改变 γ 和**禁区惩罚**
        改变参数后，下面的最优策略与状态值会**立即重算并重绘**。

        - $\gamma$ 越大越"有远见"（愿意为长期回报绕路）。
        - **禁区惩罚 $r_{\text{forbidden}}$**：这是真正能让**策略箭头改变**的旋钮。
          惩罚很轻（接近 0）时，最优策略**敢抄近路穿过禁区**；
          惩罚很重（很负）时，策略宁可**绕远路避开禁区**。
          拖动它，观察箭头整体改道——这正是赵世钰书第 3 章的经典示例。

        > 注：本 grid world 的最优策略对 γ 几乎不敏感（各 γ 下最优路径相同），
        > 所以单看 γ 时箭头基本不动、只有数值在变；禁区惩罚才会改变路线本身。
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


@app.cell
def _(P, R, greedy_policy_from_q, np, q_from_v):
    def solve_boe_history(P, R, gamma, tol=1e-9, max_iter=300):
        """返回每次迭代的 (v_k, greedy 策略_k)，最后一项是收敛解。"""
        v = np.zeros(R.shape[0])
        history = []
        for _ in range(max_iter):
            q = R + gamma * P.dot(v)
            history.append((v.copy(), greedy_policy_from_q(q)))
            v_new = q.max(axis=1)
            if np.max(np.abs(v_new - v)) < tol:
                v = v_new
                break
            v = v_new
        q = R + gamma * P.dot(v)
        history.append((v.copy(), greedy_policy_from_q(q)))   # 收敛后的最终帧
        return history
    return (solve_boe_history,)


@app.cell
def _(P, R, gamma_slider, solve_boe_history):
    boe_history = solve_boe_history(P, R, gamma_slider.value)
    n_iter = len(boe_history) - 1
    return boe_history, n_iter


@app.cell
def _(boe_history, env, gamma_slider, n_iter, rforbid_slider):
    v_final, pi_final = boe_history[-1]
    env.plot_policy(
        pi_final, v_final,
        title=(f"最优策略 π*  (γ={gamma_slider.value:.2f}, "
               f"禁区惩罚={rforbid_slider.value:.1f}, {n_iter} 步收敛)"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 🎛️ 交互区 2：逐步回放——拖动"迭代步 k"看 v 和策略怎么长出来
        从 $v_0=0$ 开始，拖动下面的滑块，观察**每一次迭代**后：
        - 格子里的数字（状态值 $v_k$）如何被一步步抬高；
        - 绿色箭头（贪心策略 $\pi_k$）如何从混乱逐渐稳定到最优。

        （此滑块只是查看预先算好的快照，拖动很快；改上面的 γ 会重新生成整段历史。）
        """
    )
    return


@app.cell
def _(mo, n_iter):
    step_slider = mo.ui.slider(
        start=0, stop=n_iter, step=1, value=0,
        label="迭代步 k", show_value=True, full_width=True,
    )
    step_slider
    return (step_slider,)


@app.cell
def _(boe_history, env, gamma_slider, np, step_slider):
    k = min(step_slider.value, len(boe_history) - 1)
    v_k, pi_k = boe_history[k]
    # 与上一步相比的最大变化量，量化"这一步动了多少"
    if k > 0:
        delta = np.max(np.abs(v_k - boe_history[k - 1][0]))
        sub = f"较上一步最大变化 Δ={delta:.3f}"
    else:
        sub = "初始 v_0 = 0（全 0，策略随意）"
    env.plot_policy(
        pi_k, v_k,
        title=f"第 {k} 步迭代  (γ={gamma_slider.value:.2f})\n{sub}",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 收敛速度：压缩映射保证 $\|v_k-v^*\|\le\gamma^k\|v_0-v^*\|$
        下图把每步到 $v^*$ 的真实距离与理论上界叠在一起（随 γ 滑块联动）。
        """
    )
    return


@app.cell
def _(boe_history, gamma_slider, np):
    import matplotlib.pyplot as plt
    v_star = boe_history[-1][0]
    dists = np.array([np.max(np.abs(vk - v_star)) for (vk, _) in boe_history[:-1]])
    dists = np.maximum(dists, 1e-12)
    bound = gamma_slider.value ** np.arange(len(dists)) * dists[0]

    figd, axd = plt.subplots(figsize=(5.4, 3.2))
    axd.semilogy(dists, "o-", ms=3, label=r"实际 $\|v_k-v^*\|_\infty$")
    axd.semilogy(bound, "--", label=r"理论上界 $\gamma^k\|v_0-v^*\|$")
    axd.set_xlabel("迭代步 k")
    axd.set_ylabel("到最优值的距离")
    axd.legend()
    axd.grid(True, alpha=0.3)
    axd.set_title(f"几何收敛 (γ={gamma_slider.value:.2f})")
    figd
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 小结
        - BOE：$v^*=\max_\pi(r_\pi+\gamma P_\pi v^*)$，解唯一且对应最优策略。
        - 右端算子是 $\gamma$-压缩 ⟹ 值迭代必然收敛；$\gamma$ 越大收敛越慢但越有远见。
        - 你已经能**逐步回放**整个求解过程了。下一章把它整理成两个经典算法：
          **值迭代** 与 **策略迭代**，并同样支持逐步查看。
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
