import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 第 8 章 · 值函数近似 (Value Function Approximation)

        前面都是**表格**：每个状态/动作存一个数。状态一多就存不下、也无法泛化。
        本章把值函数换成**参数化函数** $\hat v(s;w)\approx v_\pi(s)$，
        把"更新表格里某个数"变成"用梯度更新参数 $w$"。

        - **线性 TD**：$\hat v(s;w)=\phi(s)^\top w$，目标函数
          $J(w)=\mathbb{E}\big[(v_\pi(S)-\hat v(S;w))^2\big]$，
          用第 6 章的 SGD/随机近似来最小化。更新式：
          $$ w\leftarrow w+\alpha\big[r+\gamma\,\phi(s')^\top w-\phi(s)^\top w\big]\phi(s). $$
        - **Deep Q-Network (DQN)**：用神经网络表示 $\hat q(s,a;w)$，并引入
          **经验回放** + **目标网络**两大稳定技巧来做 Q-learning。
        """
    )
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
def _(P, R, gamma, np):
    def value_iteration(P, R, gamma, tol=1e-10):
        v = np.zeros(R.shape[0])
        while True:
            q = R + gamma * P.dot(v)
            vn = q.max(1)
            if np.max(np.abs(vn - v)) < tol:
                return vn
            v = vn
    v_opt = value_iteration(P, R, gamma)
    return (v_opt,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ## Part A · 线性 TD 评估状态值
        先设计**特征** $\phi(s)$。这里用简单的多项式特征
        $\phi(s)=[1,\ x,\ y,\ x^2,\ y^2,\ xy]$（6 维），用它去拟合"朝目标走"策略的真实状态值。
        只有 6 个参数却要表示 25 个状态 → 强迫泛化。
        """
    )
    return


@app.cell
def _(env, np):
    def feature(env, s):
        x, y = env.xy_of(s)
        x = x / (env.cols - 1)            # 归一化到 [0,1]
        y = y / (env.rows - 1)
        return np.array([1.0, x, y, x * x, y * y, x * y])

    Phi = np.stack([feature(env, s) for s in range(env.n_states)])  # (25, 6)
    print("特征矩阵 Phi:", Phi.shape)
    return Phi, feature


@app.cell
def _(P, R, env, gamma, np):
    from gridworld import ACTIONS

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

    eval_policy = toward_target_policy(env)
    P_pi = np.einsum("sa,sax->sx", eval_policy, P)
    r_pi = np.einsum("sa,sa->s", eval_policy, R)
    v_true = np.linalg.solve(np.eye(env.n_states) - gamma * P_pi, r_pi)
    return eval_policy, v_true


@app.cell
def _(env, feature, gamma, np):
    def linear_td(env, policy, feature, gamma, alpha=0.01, steps=300000, seed=0,
                  n_snapshots=40):
        rng = np.random.default_rng(seed)
        w = np.zeros(feature(env, 0).shape)
        s = env.reset(rng.integers(env.n_states))
        snaps, snap_at = [], set(
            int(x) for x in np.linspace(0, steps, n_snapshots, endpoint=False))
        for t in range(steps):
            if t in snap_at:
                snaps.append((t, w.copy()))
            a = rng.choice(env.n_actions, p=policy[s])
            ns, r, _, _ = env.step(a)
            phi_s, phi_ns = feature(env, s), feature(env, ns)
            delta = r + gamma * phi_ns @ w - phi_s @ w
            w += alpha * delta * phi_s
            s = ns
            if t % 3000 == 0:
                s = env.reset(rng.integers(env.n_states))
        snaps.append((steps, w.copy()))
        return w, snaps
    return (linear_td,)


@app.cell
def _(env, eval_policy, feature, gamma, linear_td):
    _, lin_snaps = linear_td(env, eval_policy, feature, gamma, alpha=0.01,
                             steps=300000, n_snapshots=40)
    return (lin_snaps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"### 🎛️ 逐步回放：拖动滑块看 6 个参数如何把值「拟合」出来")
    return


@app.cell
def _(lin_snaps, mo):
    lin_slider = mo.ui.slider(0, len(lin_snaps) - 1, step=1, value=0,
                              label="训练快照（越往右训练越久）", show_value=True,
                              full_width=True)
    lin_slider
    return (lin_slider,)


@app.cell
def _(Phi, env, lin_slider, lin_snaps, np, v_true):
    import matplotlib.pyplot as plt
    lin_idx = min(lin_slider.value, len(lin_snaps) - 1)
    step_at, w_lin = lin_snaps[lin_idx]
    v_approx = Phi @ w_lin
    rmse = np.sqrt(np.mean((v_approx - v_true) ** 2))
    figA, axA = plt.subplots(1, 2, figsize=(9, 4))
    for ax, vals, ttl in [(axA[0], v_true, "真实 v_π (闭式解)"),
                          (axA[1], v_approx, f"线性近似 @ {step_at} 步")]:
        env._base_ax(ax, title=ttl)
        for s in range(env.n_states):
            x, y = env.xy_of(s)
            ax.text(x, y, f"{vals[s]:.1f}", ha="center", va="center", fontsize=8)
    figA.suptitle(f"线性 TD 拟合  RMSE = {rmse:.3f}", y=1.02)
    figA
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        从左拖到右：右图数字从全 0 逐渐逼近左图真值，RMSE 同步下降。
        6 个参数就抓住了"越靠近目标值越高"的大趋势（细节有偏差，因为函数族表达力有限）。
        这就是函数近似的**泛化**威力与代价。

        ## Part B · Deep Q-Network (DQN)
        用一个小神经网络 $\hat q(s,\cdot;w)$（输入 one-hot 状态，输出 5 个动作的 Q 值）做
        Q-learning。两个关键稳定技巧：
        1. **经验回放 (replay buffer)**：把转移 $(s,a,r,s')$ 存起来随机采样，打破样本相关性。
        2. **目标网络 (target net)**：用一份滞后参数 $w^-$ 算 TD target，缓解自举发散。

        损失：$\;L(w)=\big(r+\gamma\max_{a'}\hat q(s',a';w^-)-\hat q(s,a;w)\big)^2.$
        """
    )
    return


@app.cell
def _(env, np):
    import torch
    import torch.nn as nn

    torch.manual_seed(0)

    def one_hot(s, n):
        v = torch.zeros(n)
        v[s] = 1.0
        return v

    class QNet(nn.Module):
        def __init__(self, n_states, n_actions, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_states, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x):
            return self.net(x)
    return QNet, nn, one_hot, torch


@app.cell
def _(QNet, env, np, one_hot, torch):
    import random as _random
    from collections import deque

    def train_dqn(env, gamma, episodes=400, max_steps=100, buffer_size=5000,
                  batch=64, lr=1e-3, eps_start=1.0, eps_end=0.05,
                  target_sync=200, seed=0, n_snapshots=40):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        q = QNet(nS, nA)
        q_target = QNet(nS, nA)
        q_target.load_state_dict(q.state_dict())
        opt = torch.optim.Adam(q.parameters(), lr=lr)
        buf = deque(maxlen=buffer_size)
        S = torch.eye(nS)                       # one-hot 表，行 s 即 phi(s)
        step_count = 0
        returns = []
        snaps, snap_at = [], set(
            int(x) for x in np.linspace(0, episodes, n_snapshots, endpoint=False))
        for ep in range(episodes):
            if ep in snap_at:
                with torch.no_grad():
                    snaps.append((ep, q(S).numpy()))
            eps = max(eps_end, eps_start - ep / (episodes * 0.7))
            s = env.reset(int(rng.integers(env.n_states)))   # exploring start
            ep_ret = 0.0
            for t in range(max_steps):
                if rng.random() < eps:
                    a = int(rng.integers(nA))
                else:
                    with torch.no_grad():
                        a = int(q(S[s]).argmax())
                ns, r, done, _ = env.step(a)
                buf.append((s, a, r, ns, done))
                s = ns
                ep_ret += r
                step_count += 1
                # 学习
                if len(buf) >= batch:
                    idx = rng.integers(0, len(buf), size=batch)
                    b = [buf[i] for i in idx]
                    bs = torch.stack([S[x[0]] for x in b])
                    ba = torch.tensor([x[1] for x in b])
                    br = torch.tensor([x[2] for x in b], dtype=torch.float32)
                    bns = torch.stack([S[x[3]] for x in b])
                    bd = torch.tensor([float(x[4]) for x in b])
                    qsa = q(bs).gather(1, ba[:, None]).squeeze(1)
                    with torch.no_grad():
                        target = br + gamma * (1 - bd) * q_target(bns).max(1).values
                    loss = ((qsa - target) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if step_count % target_sync == 0:
                    q_target.load_state_dict(q.state_dict())
                if done:
                    break
            returns.append(ep_ret)
        with torch.no_grad():
            snaps.append((episodes, q(S).numpy()))
        return snaps, returns
    return (train_dqn,)


@app.cell
def _(env, gamma, train_dqn):
    dqn_snaps, dqn_returns = train_dqn(env, gamma, episodes=400, seed=0,
                                       n_snapshots=40)
    return dqn_returns, dqn_snaps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"### 🎛️ 逐步回放：拖动滑块看 DQN 的策略一点点学出来")
    return


@app.cell
def _(dqn_snaps, mo):
    dqn_slider = mo.ui.slider(0, len(dqn_snaps) - 1, step=1, value=0,
                              label="训练快照（越往右训练越久）", show_value=True,
                              full_width=True)
    dqn_slider
    return (dqn_slider,)


@app.cell
def _(P, R, dqn_slider, dqn_snaps, env, gamma, greedy_policy_from_q, q_from_v, v_opt):
    dqn_idx = min(dqn_slider.value, len(dqn_snaps) - 1)
    ep_at, Q_dqn = dqn_snaps[dqn_idx]
    pi_dqn = greedy_policy_from_q(Q_dqn)
    pi_opt = greedy_policy_from_q(q_from_v(v_opt, P, R, gamma))
    agree = (pi_dqn.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(pi_dqn, Q_dqn.max(1),
                    title=f"DQN @ {ep_at} episodes  (与 π* 一致率 {agree:.0%})")
    return


@app.cell
def _(dqn_returns, np, plt):
    figR, axR = plt.subplots(figsize=(6, 3))
    rr = np.convolve(dqn_returns, np.ones(20) / 20, mode="valid")
    axR.plot(rr)
    axR.set_xlabel("episode")
    axR.set_ylabel("每回合总奖励 (滑动平均)")
    axR.set_title("DQN 学习曲线")
    axR.grid(True, alpha=0.3)
    figR
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"### 🎛️ 亲手调参：改学习率 / episode 数，点按钮重新训练 DQN")
    return


@app.cell
def _(mo):
    lr_slider = mo.ui.slider(1e-4, 5e-3, step=1e-4, value=1e-3,
                             label="学习率 lr", show_value=True, full_width=True)
    dqn_ep_slider = mo.ui.slider(100, 800, step=100, value=400,
                                 label="训练 episode 数", show_value=True,
                                 full_width=True)
    dqn_run_btn = mo.ui.run_button(label="▶ 用以上参数重新训练 DQN")
    mo.vstack([lr_slider, dqn_ep_slider, dqn_run_btn])
    return dqn_ep_slider, dqn_run_btn, lr_slider


@app.cell
def _(P, R, dqn_ep_slider, dqn_run_btn, env, gamma, greedy_policy_from_q,
      lr_slider, mo, q_from_v, train_dqn, v_opt):
    mo.stop(not dqn_run_btn.value,
            mo.md("⬆️ **调好上面的滑块后，点击「▶ 重新训练」按钮**，这里会显示训练结果。"))
    exp_snaps, _ = train_dqn(env, gamma, episodes=dqn_ep_slider.value,
                             lr=lr_slider.value, seed=7, n_snapshots=2)
    Q_exp = exp_snaps[-1][1]
    pi_exp = greedy_policy_from_q(Q_exp)
    pi_opt2 = greedy_policy_from_q(q_from_v(v_opt, P, R, gamma))
    agree_exp = (pi_exp.argmax(1) == pi_opt2.argmax(1)).mean()
    env.plot_policy(
        pi_exp, Q_exp.max(1),
        title=(f"lr={lr_slider.value:.4f}, {dqn_ep_slider.value} eps "
               f"→ 一致率 {agree_exp:.0%}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 小结
        - 函数近似把"查表"变成"拟合"，用第 6 章的随机梯度思想训练参数。
        - 线性近似简单、有理论保证；DQN 用神经网络 + **回放** + **目标网络**处理高维问题。
        - 以上仍是"**学值函数，再贪心导出策略**"。下一章换思路：
          直接对**策略**做梯度上升。
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
