import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 第 9 章 · 策略梯度方法 (Policy Gradient)

        前八章都是**先学值，再贪心导出策略**。本章直接把策略**参数化**为
        $\pi(a\mid s;\theta)$，并对目标 $J(\theta)$ 做**梯度上升**。

        ## 策略梯度定理
        $$ \nabla_\theta J(\theta)=\mathbb{E}_{s\sim d_\pi,\,a\sim\pi}
           \big[\nabla_\theta\log\pi(a|s;\theta)\;q_\pi(s,a)\big]. $$
        含义：让"带来高回报"的动作概率上升，"低回报"的下降。

        ## REINFORCE（蒙特卡洛策略梯度）
        用一条 episode 的**真实回报** $G_t$ 近似 $q_\pi(s_t,a_t)$：
        $$ \theta\leftarrow\theta+\alpha\,\gamma^t\,G_t\,\nabla_\theta\log\pi(a_t|s_t;\theta). $$
        再减去一个**基线** $b(s)$（不改变期望，只降方差）。

        策略用 **softmax**：$\pi(a|s)=\dfrac{e^{\theta_{s,a}}}{\sum_{a'}e^{\theta_{s,a'}}}$，
        此时 $\nabla_{\theta_s}\log\pi(a|s)=e_a-\pi(\cdot|s)$。
        """
    )
    return


@app.cell
def _():
    import numpy as np
    from gridworld import classic_example, greedy_policy_from_q, q_from_v
    # 策略梯度按 episode 训练，把目标设为终止态
    env = classic_example()
    P, R = env.build_model()
    gamma = env.gamma
    return P, R, env, gamma, greedy_policy_from_q, np, q_from_v


@app.cell
def _(P, R, env, gamma, greedy_policy_from_q, np, q_from_v):
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


@app.cell
def _(np):
    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def policy_from_theta(theta):
        # theta: (S, A) -> 每行做 softmax
        z = theta - theta.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    return policy_from_theta, softmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 训练：REINFORCE with baseline — 🎛️ 逐步回放
        每个 episode 从起点出发，到达目标即终止（或截断）。
        基线用状态值的简单估计 $b(s)\approx \hat v(s)$，同样在线学习。
        训练时每隔若干 episode 存一张快照，**拖动滑块**看随机策略逐渐变"尖"。
        """
    )
    return


@app.cell
def _(env, np, policy_from_theta, softmax):
    def run_episode(env, theta, max_steps, rng):
        s = env.reset(int(rng.integers(env.n_states)))   # exploring start: 覆盖全部状态
        traj = []
        for _ in range(max_steps):
            p = softmax(theta[s])
            a = rng.choice(env.n_actions, p=p)
            ns, r, done, _ = env.step(a)
            traj.append((s, a, r))
            s = ns
            if done:
                break
        return traj

    def reinforce(env, gamma, alpha_theta=0.05, alpha_v=0.1,
                  episodes=4000, max_steps=100, seed=0, n_snapshots=40):
        rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        theta = np.zeros((nS, nA))
        v = np.zeros(nS)               # 基线
        returns = []
        snaps, snap_at = [], set(
            int(x) for x in np.linspace(0, episodes, n_snapshots, endpoint=False))
        for ep in range(episodes):
            if ep in snap_at:
                snaps.append((ep, policy_from_theta(theta), v.copy()))
            traj = run_episode(env, theta, max_steps, rng)
            # 回报（逆序累加）
            G = 0.0
            Gs = []
            for (s, a, r) in reversed(traj):
                G = r + gamma * G
                Gs.append(G)
            Gs.reverse()
            ep_ret = Gs[0] if Gs else 0.0
            returns.append(ep_ret)
            # 更新
            for t, (s, a, r) in enumerate(traj):
                advantage = Gs[t] - v[s]
                v[s] += alpha_v * advantage                     # 基线向回报靠拢
                p = softmax(theta[s])
                grad_log = -p
                grad_log[a] += 1.0                              # e_a - pi
                theta[s] += alpha_theta * (gamma ** t) * advantage * grad_log
        snaps.append((episodes, policy_from_theta(theta), v.copy()))
        return theta, v, returns, snaps
    return (reinforce,)


@app.cell
def _(env, gamma, reinforce):
    _, _, pg_returns, pg_snaps = reinforce(env, gamma, episodes=4000, seed=1,
                                           n_snapshots=40)
    return pg_returns, pg_snaps


@app.cell
def _(mo, pg_snaps):
    pg_slider = mo.ui.slider(0, len(pg_snaps) - 1, step=1, value=0,
                             label="训练快照（越往右训练越久）", show_value=True,
                             full_width=True)
    pg_slider
    return (pg_slider,)


@app.cell
def _(env, pg_slider, pg_snaps, pi_opt):
    pg_idx = min(pg_slider.value, len(pg_snaps) - 1)
    ep_at, pi_pg, v_base = pg_snaps[pg_idx]
    agree = (pi_pg.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(pi_pg, v_base,
                    title=f"REINFORCE @ {ep_at} episodes (贪心与 π* 一致率 {agree:.0%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        注意策略梯度学到的是**随机策略**（每格可能有多个箭头），
        且会随训练逐渐变"尖"（接近确定性）。基线值 $\hat v(s)$ 显示在格子上方。

        ## 🎛️ 亲手调参：改 α_θ、episode 数，点按钮重新训练
        - **α_θ（策略学习率）**：太小学得慢，太大策略震荡。
        - **基线**让梯度方差更小、训练更稳。
        """
    )
    return


@app.cell
def _(mo):
    atheta_slider = mo.ui.slider(0.01, 0.3, step=0.01, value=0.05,
                                 label="策略学习率 α_θ", show_value=True,
                                 full_width=True)
    pg_ep_slider = mo.ui.slider(1000, 10000, step=1000, value=4000,
                                label="训练 episode 数", show_value=True,
                                full_width=True)
    pg_run_btn = mo.ui.run_button(label="▶ 用以上参数重新训练 REINFORCE")
    mo.vstack([atheta_slider, pg_ep_slider, pg_run_btn])
    return atheta_slider, pg_ep_slider, pg_run_btn


@app.cell
def _(atheta_slider, env, gamma, mo, pg_ep_slider, pg_run_btn, pi_opt, reinforce):
    mo.stop(not pg_run_btn.value,
            mo.md("⬆️ **调好上面的滑块后，点击「▶ 重新训练」按钮**，这里会显示训练结果。"))
    _, _, _, exp_snaps = reinforce(env, gamma, alpha_theta=atheta_slider.value,
                                   episodes=pg_ep_slider.value, seed=7, n_snapshots=2)
    _, pi_exp, v_exp = exp_snaps[-1]
    agree_exp = (pi_exp.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(
        pi_exp, v_exp,
        title=(f"α_θ={atheta_slider.value:.2f}, {pg_ep_slider.value} eps "
               f"→ 一致率 {agree_exp:.0%}"))
    return


@app.cell
def _(np, pg_returns):
    import matplotlib.pyplot as plt
    figR, axR = plt.subplots(figsize=(6, 3))
    rr = np.convolve(pg_returns, np.ones(50) / 50, mode="valid")
    axR.plot(rr)
    axR.set_xlabel("episode")
    axR.set_ylabel("每回合折扣回报 (滑动平均)")
    axR.set_title("REINFORCE 学习曲线")
    axR.grid(True, alpha=0.3)
    figR
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 小结
        - 策略梯度**直接优化策略**，天然输出随机策略，适合连续/大动作空间。
        - REINFORCE 用整条 episode 的回报 $G_t$ 当作 $q_\pi$ 的无偏估计 → 方差大。
        - **基线**能显著降方差却不引入偏差。
        - 但 REINFORCE 仍需等 episode 结束。把基线升级成**自举的 critic**，
          就得到下一章的 **Actor-Critic**：actor 出动作，critic 在线评估。
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
