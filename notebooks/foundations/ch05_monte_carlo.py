import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 5 章 · 蒙特卡洛方法 (Monte Carlo)

    从这里开始**不再需要模型** $P,R$：直接用采样到的轨迹估计动作值
    $q(s,a)=\mathbb{E}[G_t\mid S_t=s,A_t=a]$，再做策略改进。核心思想一句话：

    > 期望算不了，就用**大量样本的平均**去近似。

    本章三个算法层层递进：
    1. **MC Basic**：把策略迭代里的"评估"换成蒙特卡洛平均。
    2. **MC Exploring Starts**：用每条轨迹里的所有 (s,a) 对，数据效率更高。
    3. **MC ε-greedy**：去掉"探索性出发"这一不现实假设，用软策略保证探索。

    本章重点：用**逐步回放滑块**看 MC 的策略如何从混乱**学**到最优；
    再用**ε 滑块 + 训练按钮**亲手看探索率对结果的影响。
    """)
    return


@app.cell
def _():
    import numpy as np
    from gridworld import classic_example, greedy_policy_from_q, q_from_v
    env = classic_example()
    P, R = env.build_model()        # 仅用来算"真值"做对照，算法本身不碰它
    gamma = env.gamma
    return P, R, env, gamma, greedy_policy_from_q, np, q_from_v


@app.cell
def _(P, R, env, gamma, greedy_policy_from_q, np, q_from_v):
    # 用第 4 章的值迭代求"真·最优"，仅作为对照基准
    def value_iteration(P, R, gamma, tol=1e-10):
        v = np.zeros(R.shape[0])
        while True:
            q = R + gamma * P.dot(v)
            v_new = q.max(1)
            if np.max(np.abs(v_new - v)) < tol:
                return v_new
            v = v_new
    v_opt = value_iteration(P, R, gamma)
    pi_opt = greedy_policy_from_q(q_from_v(v_opt, P, R, gamma))
    fig_opt = env.plot_policy(pi_opt, v_opt, title="对照基准：真·最优策略 π*")
    fig_opt
    return (pi_opt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 公共部件：采样一条 episode
    给定 (s,a) 起点，按策略 $\pi$ 走 `length` 步，记录 $(s,a,r)$ 序列。
    grid world 是持续任务（目标格有自环），我们用固定步长截断。
    """)
    return


@app.cell
def _():
    def sample_episode(env, policy, s0, a0, length, rng):
        s = env.reset(s0)
        a = a0
        episode = []
        for _ in range(length):
            ns, r, _, _ = env.step(a)
            episode.append((s, a, r))
            s = ns
            a = rng.choice(env.n_actions, p=policy[s])
        return episode

    def returns_from_episode(episode, gamma):
        """逆序累加，返回与 episode 等长的每步回报 G_t。"""
        G, out = 0.0, []
        for (s, a, r) in reversed(episode):
            G = r + gamma * G
            out.append((s, a, G))
        out.reverse()
        return out

    return returns_from_episode, sample_episode


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 算法 1：MC Basic
    就是策略迭代，但 $q_\pi(s,a)$ 不再用模型算，而是从**每个 (s,a) 出发**采若干条
    episode 求平均回报。直接、好懂，但数据效率低。
    """)
    return


@app.cell
def _(greedy_policy_from_q, np, returns_from_episode, sample_episode):
    def mc_basic(env, gamma, episodes_per_sa=1, ep_len=30, iterations=20, seed=0):
        rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        pi = np.ones((nS, nA)) / nA
        Q = np.zeros((nS, nA))
        for _ in range(iterations):
            for s in range(nS):
                for a in range(nA):
                    g_sum = 0.0
                    for _ in range(episodes_per_sa):
                        ep = sample_episode(env, pi, s, a, ep_len, rng)
                        g_sum += returns_from_episode(ep, gamma)[0][2]
                    Q[s, a] = g_sum / episodes_per_sa
            pi = greedy_policy_from_q(Q)
        return pi, Q

    return (mc_basic,)


@app.cell
def _(env, gamma, mc_basic, pi_opt):
    pi_mcb, _ = mc_basic(env, gamma, episodes_per_sa=2, ep_len=30, iterations=15, seed=1)
    agree = (pi_mcb.argmax(1) == pi_opt.argmax(1)).mean()
    fig_mcb = env.plot_policy(pi_mcb, title=f"MC Basic 学到的策略 (与 π* 一致率 {agree:.0%})")
    fig_mcb
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 算法 2：MC Exploring Starts — 🎛️ 逐步回放
    一条轨迹里其实出现了很多 (s,a) 对，全都拿来更新（**first-visit**），数据利用率大增。
    "Exploring Starts" = 每条 episode 的起点 (s,a) 随机均匀，保证每个 (s,a) 都被访问。
    用增量均值在线更新：$Q\leftarrow Q+\frac1n(G-Q)$。

    训练时每隔若干 episode 存一张快照，**拖动下面的滑块**看策略一点点学出来。
    """)
    return


@app.cell
def _(greedy_policy_from_q, np, returns_from_episode, sample_episode):
    def mc_exploring_starts(env, gamma, n_episodes=20000, ep_len=30, seed=0,
                            n_snapshots=40):
        rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        Q = np.zeros((nS, nA))
        cnt = np.zeros((nS, nA))
        pi = np.ones((nS, nA)) / nA
        snaps, snap_at = [], set(
            int(x) for x in np.linspace(0, n_episodes, n_snapshots, endpoint=False))
        for ep in range(n_episodes):
            if ep in snap_at:
                snaps.append((ep, Q.copy()))
            s0 = rng.integers(nS)
            a0 = rng.integers(nA)
            episode = sample_episode(env, pi, s0, a0, ep_len, rng)
            G_list = returns_from_episode(episode, gamma)
            seen = set()
            for (s, a, G) in G_list:           # first-visit
                if (s, a) in seen:
                    continue
                seen.add((s, a))
                cnt[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / cnt[s, a]
                pi[s] = 0.0
                pi[s, Q[s].argmax()] = 1.0     # 就地贪心改进
        snaps.append((n_episodes, Q.copy()))
        return greedy_policy_from_q(Q), Q, snaps

    return (mc_exploring_starts,)


@app.cell
def _(env, gamma, mc_exploring_starts):
    _, _, es_snaps = mc_exploring_starts(env, gamma, n_episodes=20000,
                                         ep_len=30, seed=2, n_snapshots=40)
    return (es_snaps,)


@app.cell
def _(es_snaps, mo):
    es_slider = mo.ui.slider(0, len(es_snaps) - 1, step=1, value=0,
                             label="训练快照（越往右训练越久）", show_value=True,
                             full_width=True)
    es_slider
    return (es_slider,)


@app.cell
def _(env, es_slider, es_snaps, greedy_policy_from_q, pi_opt):
    es_idx = min(es_slider.value, len(es_snaps) - 1)
    ep_at, Q_es = es_snaps[es_idx]
    pi_es = greedy_policy_from_q(Q_es)
    agree_es = (pi_es.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(pi_es, Q_es.max(1),
                    title=f"MC ES @ {ep_at} episodes  (与 π* 一致率 {agree_es:.0%})")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    把滑块从左拖到右：箭头从随机乱指逐渐对齐成通往目标、绕开禁区的最优路线。

    ## 算法 3：MC ε-greedy — 🎛️ 亲手调 ε 重新训练
    "探索性出发"在真实环境里做不到（你没法任意指定起点）。
    改用 **ε-贪心**软策略：以 $1-\varepsilon$ 选贪心动作，以 $\varepsilon$ 均匀探索，
    $$ \pi(a|s)=\begin{cases}1-\varepsilon+\varepsilon/|A| & a=\arg\max_a Q\\ \varepsilon/|A| & \text{otherwise}\end{cases} $$
    - **ε 越大**探索越充分，但最终策略越"软"（每格多个箭头、更次优）。
    - **ε 越小**策略更接近确定性，但探索不足时可能漏掉某些 (s,a)。
    """)
    return


@app.cell
def _(np, returns_from_episode, sample_episode):
    def eps_greedy_policy(Q, eps):
        nS, nA = Q.shape
        pi = np.full((nS, nA), eps / nA)
        pi[np.arange(nS), Q.argmax(1)] += 1 - eps
        return pi

    def mc_epsilon_greedy(env, gamma, eps=0.1, n_episodes=30000, ep_len=40, seed=0):
        rng = np.random.default_rng(seed)
        nS, nA = env.n_states, env.n_actions
        Q = np.zeros((nS, nA))
        cnt = np.zeros((nS, nA))
        pi = eps_greedy_policy(Q, eps)
        for _ in range(n_episodes):
            s0 = rng.integers(nS)
            a0 = rng.integers(nA)
            episode = sample_episode(env, pi, s0, a0, ep_len, rng)
            seen = set()
            for (s, a, G) in returns_from_episode(episode, gamma):
                if (s, a) in seen:
                    continue
                seen.add((s, a))
                cnt[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / cnt[s, a]
            pi = eps_greedy_policy(Q, eps)
        return pi, Q

    return (mc_epsilon_greedy,)


@app.cell
def _(mo):
    eps_slider = mo.ui.slider(0.0, 0.6, step=0.02, value=0.1,
                              label="探索率 ε", show_value=True, full_width=True)
    ep_slider = mo.ui.slider(5000, 50000, step=5000, value=30000,
                             label="训练 episode 数", show_value=True, full_width=True)
    run_btn = mo.ui.run_button(label="▶ 用以上参数重新训练 MC ε-greedy")
    mo.vstack([eps_slider, ep_slider, run_btn])
    return ep_slider, eps_slider, run_btn


@app.cell
def _(
    env,
    ep_slider,
    eps_slider,
    gamma,
    mc_epsilon_greedy,
    mo,
    pi_opt,
    run_btn,
):
    mo.stop(not run_btn.value,
            mo.md("⬆️ **调好 ε 与 episode 数后，点击「▶ 重新训练」按钮**，这里会显示训练结果。"))
    pi_eg, Q_eg = mc_epsilon_greedy(env, gamma, eps=eps_slider.value,
                                    n_episodes=ep_slider.value, seed=3)
    greedy_match = (Q_eg.argmax(1) == pi_opt.argmax(1)).mean()
    env.plot_policy(
        pi_eg, Q_eg.max(1),
        title=(f"MC ε-greedy (ε={eps_slider.value:.2f}, "
               f"{ep_slider.value} eps) → 贪心一致率 {greedy_match:.0%}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ε-greedy 策略每格有多个箭头（探索），但其**贪心方向**已逼近 $\pi^*$。
    ε 越大探索越充分但最终策略越"软"（次优）；实践中常让 ε 逐渐衰减。

    ## 小结
    - MC 用样本平均代替期望，**完全不需要模型**。
    - 三个版本的演进核心是**如何保证探索**：随机起点 → exploring starts → ε-greedy。
    - 缺点：必须等一条 episode 走完才能更新。下一章的随机近似理论，
      会引出能**逐步、在线**更新的 TD 方法。
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
