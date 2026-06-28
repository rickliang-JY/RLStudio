import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 第 10 章 · Actor-Critic 方法

    Actor-Critic = **策略梯度 (actor)** + **值函数近似 (critic)**，是全书的汇合点：

    - **Actor（演员）**：参数化策略 $\pi(a|s;\theta)$，负责选动作；按策略梯度更新。
    - **Critic（评论家）**：估计值函数 $\hat v(s;w)$ 或 $\hat q(s,a;w)$，
      用 TD（第 7 章）在线学习，给 actor 提供低方差的评价信号。

    相比 REINFORCE 用整条 episode 的 $G_t$，AC 用 critic 的**自举**估计 → 可在线、每步更新、方差更低。

    本章实现两个：
    1. **QAC**：critic 学 $q$，actor 用 $q$ 当权重。
    2. **A2C (Advantage Actor-Critic)**：用 **TD error** $\delta=r+\gamma\hat v(s')-\hat v(s)$
       作为优势的无偏估计，是最常用的形式。
    """)
    return


@app.cell
def _():
    import numpy as np
    from gridworld import classic_example, greedy_policy_from_q, q_from_v
    env = classic_example()
    P, R = env.build_model()
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 算法 1：QAC (Q Actor-Critic) — 🎛️ 逐步回放
    每一步 $(s,a,r,s',a')$：
    - **Critic** 用 Sarsa 更新 $q$：$q(s,a)\leftarrow q(s,a)+\alpha_w[r+\gamma q(s',a')-q(s,a)]$。
    - **Actor** 用策略梯度更新：$\theta_s\leftarrow\theta_s+\alpha_\theta\,q(s,a)\,(e_a-\pi(\cdot|s))$。

    拖动滑块看 actor 的策略与 critic 的值一起收敛。
    """)
    return


@app.cell
def _(np, policy_from_theta, softmax):
    def qac(env, gamma, alpha_theta=0.02, alpha_w=0.1,
            episodes=4000, max_steps=100, seed=0, n_snapshots=40):
        rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        theta = np.zeros((nS, nA))      # actor
        Q = np.zeros((nS, nA))          # critic
        returns = []
        snaps, snap_at = [], set(
            int(x) for x in np.linspace(0, episodes, n_snapshots, endpoint=False))
        for ep in range(episodes):
            if ep in snap_at:
                snaps.append((ep, policy_from_theta(theta), Q.max(1).copy()))
            s = env.reset(int(rng.integers(nS)))   # exploring start
            a = rng.choice(nA, p=softmax(theta[s]))
            ep_ret, disc = 0.0, 1.0
            for t in range(max_steps):
                ns, r, done, _ = env.step(a)
                na = rng.choice(nA, p=softmax(theta[ns]))
                # critic: Sarsa TD
                target = r + (0.0 if done else gamma * Q[ns, na])
                Q[s, a] += alpha_w * (target - Q[s, a])
                # actor: policy gradient，权重 = q(s,a)
                p = softmax(theta[s])
                grad_log = -p
                grad_log[a] += 1.0
                theta[s] += alpha_theta * disc * Q[s, a] * grad_log
                ep_ret += r
                disc *= gamma
                s, a = ns, na
                if done:
                    break
            returns.append(ep_ret)
        snaps.append((episodes, policy_from_theta(theta), Q.max(1).copy()))
        return theta, Q, returns, snaps

    return (qac,)


@app.cell
def _(env, gamma, qac):
    _, _, ret_qac, qac_snaps = qac(env, gamma, episodes=4000, seed=1, n_snapshots=40)
    return qac_snaps, ret_qac


@app.cell
def _(mo, qac_snaps):
    qac_slider = mo.ui.slider(0, len(qac_snaps) - 1, step=1, value=0,
                              label="QAC 训练快照（越往右训练越久）", show_value=True,
                              full_width=True)
    qac_slider
    return (qac_slider,)


@app.cell
def _(env, pi_opt, qac_slider, qac_snaps):
    qac_idx = min(qac_slider.value, len(qac_snaps) - 1)
    ep_qac, pi_qac, vq = qac_snaps[qac_idx]
    agree_qac = (pi_qac.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(pi_qac, vq,
                    title=f"QAC @ {ep_qac} episodes (贪心与 π* 一致率 {agree_qac:.0%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 算法 2：A2C (Advantage Actor-Critic) — 🎛️ 逐步回放 + 调参
    Critic 只学**状态值** $\hat v(s)$，用 **TD error** 当作优势：
    $$ \delta_t=r+\gamma\hat v(s')-\hat v(s). $$
    - **Critic**：$w$（这里表格化的 $\hat v$）$\leftarrow \hat v(s)+\alpha_w\,\delta$。
    - **Actor**：$\theta_s\leftarrow\theta_s+\alpha_\theta\,\delta\,(e_a-\pi(\cdot|s))$。

    $\delta$ 同时充当 critic 的学习信号和 actor 的优势信号——这是 A2C 的精妙之处。
    """)
    return


@app.cell
def _(np, policy_from_theta, softmax):
    def a2c(env, gamma, alpha_theta=0.02, alpha_w=0.1,
            episodes=4000, max_steps=100, seed=0, n_snapshots=40):
        rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        theta = np.zeros((nS, nA))      # actor
        V = np.zeros(nS)                # critic (状态值)
        returns = []
        snaps, snap_at = [], set(
            int(x) for x in np.linspace(0, episodes, n_snapshots, endpoint=False))
        for ep in range(episodes):
            if ep in snap_at:
                snaps.append((ep, policy_from_theta(theta), V.copy()))
            s = env.reset(int(rng.integers(nS)))   # exploring start
            ep_ret, disc = 0.0, 1.0
            for t in range(max_steps):
                p = softmax(theta[s])
                a = rng.choice(nA, p=p)
                ns, r, done, _ = env.step(a)
                # TD error = 优势的无偏估计
                target = r + (0.0 if done else gamma * V[ns])
                delta = target - V[s]
                # critic
                V[s] += alpha_w * delta
                # actor
                grad_log = -p
                grad_log[a] += 1.0
                theta[s] += alpha_theta * disc * delta * grad_log
                ep_ret += r
                disc *= gamma
                s = ns
                if done:
                    break
            returns.append(ep_ret)
        snaps.append((episodes, policy_from_theta(theta), V.copy()))
        return theta, V, returns, snaps

    return (a2c,)


@app.cell
def _(a2c, env, gamma):
    _, _, ret_a2c, a2c_snaps = a2c(env, gamma, episodes=4000, seed=1, n_snapshots=40)
    return a2c_snaps, ret_a2c


@app.cell
def _(a2c_snaps, mo):
    a2c_slider = mo.ui.slider(0, len(a2c_snaps) - 1, step=1, value=0,
                              label="A2C 训练快照（越往右训练越久）", show_value=True,
                              full_width=True)
    a2c_slider
    return (a2c_slider,)


@app.cell
def _(a2c_slider, a2c_snaps, env, pi_opt):
    a2c_idx = min(a2c_slider.value, len(a2c_snaps) - 1)
    ep_a2c, pi_a2c, V_a2c = a2c_snaps[a2c_idx]
    agree_a2c = (pi_a2c.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(pi_a2c, V_a2c,
                    title=f"A2C @ {ep_a2c} eps + critic 状态值 (一致率 {agree_a2c:.0%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🎛️ 亲手调参：改 α_θ、α_w、episode 数，点按钮重新训练 A2C
    """)
    return


@app.cell
def _(mo):
    a2c_at = mo.ui.slider(0.005, 0.2, step=0.005, value=0.02,
                          label="actor 学习率 α_θ", show_value=True, full_width=True)
    a2c_aw = mo.ui.slider(0.02, 0.5, step=0.02, value=0.1,
                          label="critic 学习率 α_w", show_value=True, full_width=True)
    a2c_ep = mo.ui.slider(1000, 10000, step=1000, value=4000,
                          label="训练 episode 数", show_value=True, full_width=True)
    a2c_run = mo.ui.run_button(label="▶ 用以上参数重新训练 A2C")
    mo.vstack([a2c_at, a2c_aw, a2c_ep, a2c_run])
    return a2c_at, a2c_aw, a2c_ep, a2c_run


@app.cell
def _(a2c, a2c_at, a2c_aw, a2c_ep, a2c_run, env, gamma, mo, pi_opt):
    mo.stop(not a2c_run.value,
            mo.md("⬆️ **调好上面的滑块后，点击「▶ 重新训练」按钮**，这里会显示训练结果。"))
    _, _, _, exp_snaps = a2c(env, gamma, alpha_theta=a2c_at.value,
                             alpha_w=a2c_aw.value, episodes=a2c_ep.value,
                             seed=7, n_snapshots=2)
    _, pi_exp, V_exp = exp_snaps[-1]
    agree_exp = (pi_exp.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(
        pi_exp, V_exp,
        title=(f"α_θ={a2c_at.value:.3f}, α_w={a2c_aw.value:.2f}, "
               f"{a2c_ep.value} eps → 一致率 {agree_exp:.0%}"))
    return


@app.cell
def _(np, ret_a2c, ret_qac):
    import matplotlib.pyplot as plt

    def smooth(x, k=50):
        return np.convolve(np.asarray(x, float), np.ones(k) / k, mode="valid")

    figC, axC = plt.subplots(figsize=(6, 3.2))
    axC.plot(smooth(ret_qac), label="QAC")
    axC.plot(smooth(ret_a2c), label="A2C (advantage)")
    axC.set_xlabel("episode")
    axC.set_ylabel("每回合总奖励 (滑动平均)")
    axC.set_title("Actor-Critic 学习曲线对比")
    axC.legend()
    axC.grid(True, alpha=0.3)
    figC
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 全书收官
    | 章 | 关键词 | 是否需要模型 |
    |---|---|---|
    | 2-4 | 贝尔曼方程 / 值迭代 / 策略迭代 | ✅ 需要 |
    | 5 | 蒙特卡洛 | ❌ |
    | 6 | 随机近似（数学工具） | — |
    | 7 | TD / Sarsa / Q-learning | ❌ |
    | 8 | 值函数近似 / DQN | ❌ |
    | 9 | 策略梯度 / REINFORCE | ❌ |
    | 10 | **Actor-Critic** | ❌ |

    Actor-Critic 把"值"（critic）与"策略"（actor）两条主线合一：
    - critic 用 **TD/随机近似**（6、7、8 章）在线评估；
    - actor 用 **策略梯度**（9 章）直接优化。

    这正是现代深度强化学习（A2C/A3C、PPO、SAC、DDPG）的统一骨架。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
