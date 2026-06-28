import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 7 章 · 时序差分方法 (Temporal-Difference)

    TD = **蒙特卡洛的在线版** + **随机近似的应用**：不必等 episode 结束，
    每走一步就用**自举 (bootstrapping)** 更新——用"下一步的估计"改进"当前估计"。

    统一形式：新估计 ← 旧估计 + α·[ TD target − 旧估计 ]

    | 算法 | 学什么 | TD target | 用途 |
    |---|---|---|---|
    | **TD(0)** | $v_\pi(s)$ | $r+\gamma v(s')$ | 评估给定策略 |
    | **Sarsa** | $q_\pi(s,a)$ | $r+\gamma q(s',a')$ | on-policy 控制 |
    | **Expected Sarsa** | $q_\pi$ | $r+ \gamma \sum_{a'}\pi(a'\|s')q(s',a')$ | 方差更小 |
    | **Q-learning** | $q^*(s,a)$ | $r+\gamma\max_{a'}q(s',a')$ | off-policy 控制 |

    本章重点：用**逐步回放滑块**看 Q-learning 的策略如何从混乱**学**到最优；
    再用**参数滑块 + 训练按钮**亲手调 α、ε，看对结果的影响。
    """)
    return


@app.cell
def _():
    import numpy as np
    from gridworld import classic_example, greedy_policy_from_q, q_from_v
    env = classic_example()
    P, R = env.build_model()      # 仅用于对照真值
    gamma = env.gamma
    return P, R, env, gamma, greedy_policy_from_q, np, q_from_v


@app.cell
def _(P, R, gamma, greedy_policy_from_q, np, q_from_v):
    def value_iteration(P, R, gamma, tol=1e-10):
        v = np.zeros(R.shape[0])
        while True:
            q = R + gamma * P.dot(v)
            vn = q.max(1)
            if np.max(np.abs(vn - v)) < tol:
                return vn
            v = vn
    v_opt = value_iteration(P, R, gamma)
    pi_opt = greedy_policy_from_q(q_from_v(v_opt, P, R, gamma))
    return (pi_opt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 公共部件：带快照记录的 Q-learning
    训练时每隔若干 episode 存一张"策略 + 价值"快照，方便之后**逐步回放**学习过程。
    采用 exploring start（每个 episode 随机起点）以覆盖所有状态。
    """)
    return


@app.cell
def _(np):
    def eps_greedy_action(Q, s, eps, rng):
        if rng.random() < eps:
            return int(rng.integers(Q.shape[1]))
        return int(Q[s].argmax())

    def q_learning(env, gamma, alpha=0.1, eps=0.1, episodes=5000, max_steps=200,
                   seed=0, n_snapshots=40):
        rng = np.random.default_rng(seed)
        Q = np.zeros((env.n_states, env.n_actions))
        lengths = []
        snaps, snap_at = [], set(
            int(x) for x in np.linspace(0, episodes, n_snapshots, endpoint=False))
        for ep in range(episodes):
            if ep in snap_at:
                snaps.append((ep, Q.copy()))
            s = env.reset(int(rng.integers(env.n_states)))
            for t in range(max_steps):
                a = eps_greedy_action(Q, s, eps, rng)
                ns, r, done, _ = env.step(a)
                Q[s, a] += alpha * (r + gamma * Q[ns].max() - Q[s, a])
                s = ns
                if done:
                    break
            lengths.append(t + 1)
        snaps.append((episodes, Q.copy()))
        return Q, lengths, snaps

    return eps_greedy_action, q_learning


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🎛️ 逐步回放：拖动滑块，看 Q-learning 的策略一点点学出来
    """)
    return


@app.cell
def _(env, gamma, q_learning):
    _, ql_lengths, ql_snaps = q_learning(env, gamma, alpha=0.1, eps=0.1,
                                         episodes=5000, seed=1, n_snapshots=40)
    return ql_lengths, ql_snaps


@app.cell
def _(mo, ql_snaps):
    snap_slider = mo.ui.slider(0, len(ql_snaps) - 1, step=1, value=0,
                               label="训练快照（越往右训练越久）", show_value=True,
                               full_width=True)
    snap_slider
    return (snap_slider,)


@app.cell
def _(env, greedy_policy_from_q, pi_opt, ql_snaps, snap_slider):
    idx = min(snap_slider.value, len(ql_snaps) - 1)
    ep_at, Q_snap = ql_snaps[idx]
    pi_snap = greedy_policy_from_q(Q_snap)
    agree = (pi_snap.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(pi_snap, Q_snap.max(1),
                    title=f"Q-learning @ {ep_at} episodes  (与 π* 一致率 {agree:.0%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    把滑块从左拖到右：箭头从随机乱指逐渐对齐成通往目标、绕开禁区的最优路线，
    格子数值也同步收敛。这就是"学习"在发生。

    ## 🎛️ 亲手调参：改 α、ε，点按钮重新训练
    - **α（学习率）**：太小学得慢，太大不稳定。
    - **ε（探索率）**：太小可能探索不足陷入次优，太大学到的策略偏"软"。
    """)
    return


@app.cell
def _(mo):
    alpha_slider = mo.ui.slider(0.01, 0.9, step=0.01, value=0.1,
                                label="学习率 α", show_value=True, full_width=True)
    eps_slider = mo.ui.slider(0.0, 0.6, step=0.02, value=0.1,
                              label="探索率 ε", show_value=True, full_width=True)
    ep_slider = mo.ui.slider(500, 8000, step=500, value=3000,
                             label="训练 episode 数", show_value=True, full_width=True)
    run_btn = mo.ui.run_button(label="▶ 用以上参数重新训练 Q-learning")
    mo.vstack([alpha_slider, eps_slider, ep_slider, run_btn])
    return alpha_slider, ep_slider, eps_slider, run_btn


@app.cell
def _(
    alpha_slider,
    env,
    ep_slider,
    eps_slider,
    gamma,
    greedy_policy_from_q,
    mo,
    pi_opt,
    q_learning,
    run_btn,
):
    mo.stop(not run_btn.value,
            mo.md("⬆️ **调好上面的滑块后，点击「▶ 重新训练」按钮**，这里会显示训练结果。"))
    Q_exp, _, _ = q_learning(env, gamma, alpha=alpha_slider.value,
                             eps=eps_slider.value, episodes=ep_slider.value, seed=7)
    pi_exp = greedy_policy_from_q(Q_exp)
    agree_exp = (pi_exp.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(
        pi_exp, Q_exp.max(1),
        title=(f"α={alpha_slider.value:.2f}, ε={eps_slider.value:.2f}, "
               f"{ep_slider.value} eps → 一致率 {agree_exp:.0%}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sarsa vs Q-learning：学习曲线
    Sarsa 是 on-policy（学"含探索的策略"的值），Q-learning 是 off-policy（直接学 $q^*$）。
    """)
    return


@app.cell
def _(eps_greedy_action, np):
    def sarsa(env, gamma, alpha=0.1, eps=0.1, episodes=5000, max_steps=200, seed=0):
        rng = np.random.default_rng(seed)
        Q = np.zeros((env.n_states, env.n_actions))
        lengths = []
        for _ in range(episodes):
            s = env.reset(int(rng.integers(env.n_states)))
            a = eps_greedy_action(Q, s, eps, rng)
            for t in range(max_steps):
                ns, r, done, _ = env.step(a)
                na = eps_greedy_action(Q, ns, eps, rng)
                Q[s, a] += alpha * (r + gamma * Q[ns, na] - Q[s, a])
                s, a = ns, na
                if done:
                    break
            lengths.append(t + 1)
        return Q, lengths

    return (sarsa,)


@app.cell
def _(env, gamma, sarsa):
    _, sarsa_lengths = sarsa(env, gamma, episodes=5000, seed=8)
    return (sarsa_lengths,)


@app.cell
def _(np, ql_lengths, sarsa_lengths):
    import matplotlib.pyplot as plt

    def smooth(x, k=100):
        return np.convolve(np.asarray(x, float), np.ones(k) / k, mode="valid")

    figL, axL = plt.subplots(figsize=(6, 3.2))
    axL.plot(smooth(sarsa_lengths), label="Sarsa")
    axL.plot(smooth(ql_lengths), label="Q-learning")
    axL.set_xlabel("episode")
    axL.set_ylabel("到达目标步数 (滑动平均)")
    axL.set_title("学习曲线：步数下降 = 策略变好")
    axL.legend()
    axL.grid(True, alpha=0.3)
    figL
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 小结
    - TD 在线、无模型地学 $v$ 或 $q$；**on-policy (Sarsa)** vs **off-policy (Q-learning)**。
    - 你已经能**逐步回放**学习过程、并**亲手调参**重训。
    - 以上是**表格型**（每个 $(s,a)$ 存一个数）。状态一多就存不下，
      下一章用**函数近似**把表格换成参数化函数。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
