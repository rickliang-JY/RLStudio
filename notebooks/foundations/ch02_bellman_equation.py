import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 2 章 · 状态值与贝尔曼方程 (State Values & Bellman Equation)

    给定策略 $\pi$，它"有多好"？答案是**状态值**
    $v_\pi(s)=\mathbb{E}[G_t\mid S_t=s]$。
    它满足**贝尔曼方程**

    $v_\pi(s)=\sum_a \pi(a|s)\Big(r(s,a)+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')\Big),$
    $\qquad v_\pi=r_\pi+\gamma P_\pi v_\pi.$

    本章用**两种解法**（闭式 / 迭代）求状态值，并让你**交互地**改 γ、换策略，
    还能**逐步回放**迭代解的收敛过程。
    """)
    return


@app.cell
def _():
    import numpy as np
    from gridworld import classic_example, ACTIONS
    env = classic_example()
    P, R = env.build_model()
    return ACTIONS, P, R, env, np


@app.cell
def _(ACTIONS, env, np):
    def random_policy(env):
        return np.ones((env.n_states, env.n_actions)) / env.n_actions

    def toward_target_policy(env):
        pi = np.zeros((env.n_states, env.n_actions))
        tx, ty = env.target
        for s in range(env.n_states):
            x, y = env.xy_of(s)
            best_a, best_d = 4, abs(x - tx) + abs(y - ty)
            for a, (dx, dy) in enumerate(ACTIONS):
                nx, ny = x + dx, y + dy
                if env.in_bounds(nx, ny) and abs(nx - tx) + abs(ny - ty) < best_d:
                    best_d, best_a = abs(nx - tx) + abs(ny - ty), a
            pi[s, best_a] = 1.0
        return pi

    POLICIES = {"均匀随机策略": random_policy(env),
                "朝目标走的策略": toward_target_policy(env)}
    return (POLICIES,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🎛️ 交互区：选策略 + 调 γ，闭式解状态值实时更新
    """)
    return


@app.cell
def _(POLICIES, mo):
    policy_dropdown = mo.ui.dropdown(options=list(POLICIES.keys()),
                                     value="朝目标走的策略", label="评估哪个策略")
    gamma_slider = mo.ui.slider(0.1, 0.99, step=0.01, value=0.9,
                                label="折扣因子 γ", show_value=True, full_width=True)
    mo.vstack([policy_dropdown, gamma_slider])
    return gamma_slider, policy_dropdown


@app.cell
def _(P, POLICIES, R, env, gamma_slider, np, policy_dropdown):
    def bellman_closed_form(policy, P, R, gamma):
        P_pi = np.einsum("sa,sax->sx", policy, P)
        r_pi = np.einsum("sa,sa->s", policy, R)
        return np.linalg.solve(np.eye(len(r_pi)) - gamma * P_pi, r_pi)

    cur_policy = POLICIES[policy_dropdown.value]
    v_closed = bellman_closed_form(cur_policy, P, R, gamma_slider.value)
    env.plot_policy(cur_policy, v_closed,
                    title=f"{policy_dropdown.value} 的状态值 (γ={gamma_slider.value:.2f})")
    return cur_policy, v_closed


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    换成"均匀随机策略"会看到值普遍更低（乱走会反复撞墙/踩禁区）；
    调大 γ 会让远处的格子也获得更高的值（更看重长期回报）。

    ## 迭代解法 + 逐步回放
    闭式解要求逆矩阵；改用**不动点迭代** $v_{k+1}=r_\pi+\gamma P_\pi v_k$，
    由 $\gamma$-压缩保证线性收敛。拖动滑块看 $v_k$ 一步步逼近真值。
    """)
    return


@app.cell
def _(P, R, cur_policy, gamma_slider, np):
    def bellman_iter_history(policy, P, R, gamma, tol=1e-9, max_iter=200):
        P_pi = np.einsum("sa,sax->sx", policy, P)
        r_pi = np.einsum("sa,sa->s", policy, R)
        v = np.zeros(len(r_pi))
        hist = [v.copy()]
        for _ in range(max_iter):
            v = r_pi + gamma * P_pi @ v
            hist.append(v.copy())
            if np.max(np.abs(hist[-1] - hist[-2])) < tol:
                break
        return hist

    iter_hist = bellman_iter_history(cur_policy, P, R, gamma_slider.value)
    n_steps = len(iter_hist) - 1
    return iter_hist, n_steps


@app.cell
def _(mo, n_steps):
    iter_slider = mo.ui.slider(0, n_steps, step=1, value=0,
                               label="迭代步 k", show_value=True, full_width=True)
    iter_slider
    return (iter_slider,)


@app.cell
def _(cur_policy, env, iter_hist, iter_slider, n_steps, np, v_closed):
    k = min(iter_slider.value, len(iter_hist) - 1)
    vk = iter_hist[k]
    err = np.max(np.abs(vk - v_closed))
    env.plot_policy(cur_policy, vk,
                    title=f"迭代解 第 {k}/{n_steps} 步  (与闭式解最大误差 {err:.3f})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结
    - 状态值是评价策略好坏的标尺；贝尔曼方程 $v_\pi=r_\pi+\gamma P_\pi v_\pi$ 有闭式/迭代两解法。
    - 迭代法是后面**值迭代、策略迭代、TD** 的雏形。
    - 下一章把"评价"升级为"优化"：在所有策略里找最好的 → 贝尔曼最优方程。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
