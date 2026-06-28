import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 第 5 章 · 书中例子复现 (Worked Examples)

    本notebook把赵世钰《强化学习的数学原理》第 5 章里出现的**经典插图例子**逐一
    复现出来，并全部做成**可交互**的：拖动滑块/换动作，立刻看到书里那张图怎么变。

    | 例子 | 书中图 | 演示什么 |
    |---|---|---|
    | 1. 大数定律估均值 | Fig 5.2 | 样本平均如何收敛到期望 |
    | 2. MC Basic 手算一步 | Fig 5.3 | 3×3 网格上 $q_{\pi_0}(s_1,\cdot)$ 怎么从一条轨迹算出来 |
    | 3. Episode 长度的影响 | Fig 5.4 | 轨迹太短 → 值/策略学不出来 |
    | 4. 最优 ε-greedy 策略 | Fig 5.7 | ε 越大值越低、且偏离最优贪心策略 |
    | 5. ε 的探索能力 | Fig 5.8 | ε=1 访问均匀；ε=0.5 访问极不均 |

    > 书中第 5 章 5×5 例子的奖励设定（与本notebook一致）：
    > $r_{\text{boundary}}=-1,\ r_{\text{forbidden}}=-10,\ r_{\text{target}}=1,\ r_{\text{other}}=0,\ \gamma=0.9$。
    > 注意 $r_{\text{forbidden}}=-10$（比前几章的 −1 更狠），这正是书里 Fig 5.7 那组数值的来源。
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from gridworld import GridWorld, greedy_policy_from_q
    return GridWorld, greedy_policy_from_q, np, plt


@app.cell
def _(GridWorld):
    # 第 5 章 5×5 例子的“官方”网格（Fig 5.4–5.8 全部用它）
    def book_grid():
        return GridWorld(
            cols=5, rows=5, start=(0, 0), target=(2, 3),
            forbidden=((1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)),
            r_target=1.0, r_forbidden=-10.0, r_step=0.0,
            r_boundary=-1.0, gamma=0.9,
        )
    return (book_grid,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 1 · 大数定律估均值 (Fig 5.2)
    抛一枚“硬币”得到随机变量 $X\in\{+1,-1\}$，$P(X=1)=p$，期望 $\mathbb{E}[X]=2p-1$。
    蒙特卡洛的根基就是：**期望算不了，就用大量样本的平均去近似**。
    拖动 $p$ 改变真值，拖动**样本数**看平均值如何一点点压到那条虚线上。
    """)
    return


@app.cell
def _(mo):
    p_slider = mo.ui.slider(0.0, 1.0, step=0.05, value=0.5,
                            label="P(X=+1) = p", show_value=True, full_width=True)
    n_slider = mo.ui.slider(10, 2000, step=10, value=200,
                            label="样本数 n（看到第几个样本）", show_value=True,
                            full_width=True)
    seed_slider = mo.ui.slider(0, 20, step=1, value=0,
                               label="随机种子", show_value=True, full_width=True)
    mo.vstack([p_slider, n_slider, seed_slider])
    return n_slider, p_slider, seed_slider


@app.cell
def _(n_slider, np, p_slider, plt, seed_slider):
    p = p_slider.value
    n = n_slider.value
    truth = 2 * p - 1
    rng = np.random.default_rng(seed_slider.value)
    x = np.where(rng.random(n) < p, 1.0, -1.0)
    avg = np.cumsum(x) / np.arange(1, n + 1)

    fig_lln, ax_lln = plt.subplots(figsize=(6.2, 3.4))
    ax_lln.scatter(np.arange(1, n + 1), x, s=6, color="gray", alpha=0.25,
                   label="每个样本 ±1")
    ax_lln.plot(np.arange(1, n + 1), avg, "-", color="tab:blue",
                label=r"样本平均 $\bar X_n$")
    ax_lln.axhline(truth, ls="--", color="tab:red",
                   label=f"真值 E[X]={truth:+.2f}")
    ax_lln.set_xlabel("样本编号 n")
    ax_lln.set_ylabel("值")
    ax_lln.set_ylim(-1.25, 1.25)
    ax_lln.set_title(f"大数定律：n={n} 时 平均={avg[-1]:+.3f}（真值 {truth:+.2f}）")
    ax_lln.legend(fontsize=8, loc="upper right")
    ax_lln.grid(True, alpha=0.3)
    fig_lln
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    样本少时平均值上下乱跳，样本一多就被“摁”在真值附近——这就是 MC 能用平均代替期望的底气。

    ## 例子 2 · MC Basic 手算一步 (Fig 5.3)
    书里用一个 **3×3** 网格、确定性策略 $\pi_0$，**手算** $q_{\pi_0}(s_1,\cdot)$。
    因为环境和策略都确定，从 $(s_1,a)$ 出发只会生成**唯一一条**轨迹，算一条就够。
    下面**选 $s_1$ 处的动作 $a$**，看那条轨迹长什么样、回报怎么逆推出 $q$ 值。
    （书中结果：$q(s_1,\cdot)=[-10,\ 7.29,\ 7.29,\ -10,\ -9]$，对应 上/右/下/左/停。）
    """)
    return


@app.cell
def _(GridWorld, np):
    # Fig 5.3 的 3×3 网格：目标 s9=(2,2)，无禁区，r_boundary=r_forbidden=-1, r_other=0
    def fig53_grid():
        return GridWorld(
            cols=3, rows=3, start=(0, 0), target=(2, 2), forbidden=(),
            r_target=1.0, r_forbidden=-1.0, r_step=0.0,
            r_boundary=-1.0, gamma=0.9,
        )

    # 书中给定的初始确定性策略 π0（动作: 0上 1右 2下 3左 4停）
    def fig53_pi0(env):
        det = {
            (0, 0): 0, (1, 0): 2, (2, 0): 2,
            (0, 1): 1, (1, 1): 2, (2, 1): 2,
            (0, 2): 1, (1, 2): 1, (2, 2): 4,
        }
        pi = np.zeros((env.n_states, env.n_actions))
        for (x, y), a in det.items():
            pi[env.s_of(x, y), a] = 1.0
        return pi
    return fig53_grid, fig53_pi0


@app.cell
def _(mo):
    a_dropdown = mo.ui.dropdown(
        options={"a1 = 上": 0, "a2 = 右": 1, "a3 = 下": 2, "a4 = 左": 3, "a5 = 停": 4},
        value="a2 = 右",
        label="在 s1 处先走哪个动作 a？")
    a_dropdown
    return (a_dropdown,)


@app.cell
def _(a_dropdown, fig53_grid, fig53_pi0, mo):
    def det_episode(env, pi, s0, a0, length=160):
        env.reset(s0)
        a = a0
        ep, traj = [], [s0]
        s = s0
        for _ in range(length):
            ns, r, _, _ = env.step(a)
            ep.append((s, a, r))
            traj.append(ns)
            s = ns
            a = int(pi[s].argmax())
        return ep, traj

    def disc_return(ep, gamma):
        G = 0.0
        for (_s, _a, r) in reversed(ep):
            G = r + gamma * G
        return G

    env53 = fig53_grid()
    pi053 = fig53_pi0(env53)
    a0 = a_dropdown.value
    ep_sel, traj_sel = det_episode(env53, pi053, 0, a0)
    G_sel = disc_return(ep_sel, env53.gamma)

    qs = []
    for aa in range(5):
        e, _ = det_episode(env53, pi053, 0, aa)
        qs.append(disc_return(e, env53.gamma))

    names = ["a1上", "a2右", "a3下", "a4左", "a5停"]
    rows = "\n".join(
        f"| {names[i]} | {qs[i]:+.2f} |" + ("  ← 你选的" if i == a0 else " |")
        for i in range(5))
    fig53 = env53.plot_trajectory(
        traj_sel[:14], mark_current=True,
        title=f"从 (s1, {names[a0]}) 出发的唯一轨迹  →  q = {G_sel:+.2f}")
    mo.hstack([
        fig53,
        mo.md(f"""**$q_{{\\pi_0}}(s_1,\\cdot)$（逆序累加折扣回报）**

| 动作 | q 值 |
|---|---|
{rows}

最大值在 **a2(右)/a3(下)** $=\\dfrac{{\\gamma^3}}{{1-\\gamma}}=7.29$，
故策略改进后 $s_1$ 应改取右或下——与书中结论一致。""")
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 例子 3 · Episode 长度的影响 (Fig 5.4)
    回到 5×5 官方网格。MC Basic 要从每个 $(s,a)$ 采一条长度固定的 episode 估 $q$。
    **如果 episode 太短**，远离目标的状态根本走不到目标、采不到那个 +1，
    于是它们的值/策略都学不出来。**拖动“episode 长度”**，看正确的策略箭头如何
    从目标附近一圈圈向外“长”出来。
    """)
    return


@app.cell
def _(book_grid, greedy_policy_from_q, np):
    def mc_basic_fixed_len(env, ep_len, iterations=40):
        nS, nA, g = env.n_states, env.n_actions, env.gamma
        pi = np.zeros((nS, nA)); pi[:, 0] = 1.0      # 确定性初始策略：全部“上”
        Q = np.zeros((nS, nA))
        for _ in range(iterations):
            for s in range(nS):
                for a in range(nA):
                    env.reset(s); aa = a; rewards = []
                    for _ in range(ep_len):
                        ns, r, _, _ = env.step(aa)
                        rewards.append(r)
                        aa = int(pi[ns].argmax())
                    G = 0.0
                    for r in reversed(rewards):
                        G = r + g * G
                    Q[s, a] = G
            pi = greedy_policy_from_q(Q)
        return pi, Q

    fig54_lengths = [1, 2, 3, 4, 10, 15, 20, 30, 50, 100]
    fig54_env = book_grid()
    fig54_results = {L: mc_basic_fixed_len(fig54_env, L) for L in fig54_lengths}
    return fig54_env, fig54_lengths, fig54_results


@app.cell
def _(fig54_lengths, mo):
    len_slider = mo.ui.slider(0, len(fig54_lengths) - 1, step=1, value=2,
                              label="episode 长度档位（越往右越长）",
                              show_value=False, full_width=True)
    len_slider
    return (len_slider,)


@app.cell
def _(fig54_env, fig54_lengths, fig54_results, len_slider):
    L = fig54_lengths[min(len_slider.value, len(fig54_lengths) - 1)]
    pi54, Q54 = fig54_results[L]
    fig54_env.plot_policy(
        pi54, Q54.max(1),
        title=f"MC Basic · episode 长度 = {L}（格上为估出的 max_a q）")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    长度从 1 往上拖：一开始只有目标周围几格学对，长度足够（书中约 ≥15）后整张策略才完全正确。
    这说明 MC 对 episode 长度敏感——太短等于看不到远方的奖励。

    ## 例子 4 · 最优 ε-greedy 策略 (Fig 5.7)
    把贪心策略换成 **ε-贪心**软策略后，“在所有 ε-贪心策略里最优”的那一个长什么样？
    拖动 **ε**，对照书 Fig 5.7(a)–(d)：
    - **ε=0**：就是普通最优策略，目标值 10.0。
    - **ε 增大**：所有状态值整体下降；当 ε 大到 0.5，连**目标格的最优动作都不再是“停”**
      （因为软策略下有较大概率乱走进周围禁区吃 −10），而是“逃离”。
    """)
    return


@app.cell
def _(book_grid, np):
    def optimal_eps_greedy(env, eps, tol=1e-10, max_iter=10000):
        P, R = env.build_model()
        nS, nA, g = env.n_states, env.n_actions, env.gamma
        v = np.zeros(nS)
        for _ in range(max_iter):
            q = R + g * P.dot(v)                 # (S, A)
            greedy = q.argmax(1)
            pi = np.full((nS, nA), eps / nA)
            pi[np.arange(nS), greedy] += 1 - eps
            vn = (pi * q).sum(1)
            if np.max(np.abs(vn - v)) < tol:
                v = vn
                break
            v = vn
        q = R + g * P.dot(v)
        greedy = q.argmax(1)
        pi = np.full((nS, nA), eps / nA)
        pi[np.arange(nS), greedy] += 1 - eps
        return pi, v
    return (optimal_eps_greedy,)


@app.cell
def _(mo):
    eps57_slider = mo.ui.slider(0.0, 0.5, step=0.05, value=0.0,
                                label="探索率 ε（对照书 Fig 5.7 的 0 / 0.1 / 0.2 / 0.5）",
                                show_value=True, full_width=True)
    eps57_slider
    return (eps57_slider,)


@app.cell
def _(book_grid, eps57_slider, optimal_eps_greedy):
    env57 = book_grid()
    pi57, v57 = optimal_eps_greedy(env57, eps57_slider.value)
    env57.plot_policy(
        pi57, v57,
        title=f"最优 ε-greedy 策略 (ε={eps57_slider.value:.2f})  目标值={v57[env57.s_of(2, 3)]:.1f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    把 ε 拖到 0.5：每格出现多个箭头（探索），整体数值大幅变负，目标格的值反而最小。
    这正是书里的结论——**ε 太大，最优 ε-贪心策略会偏离真正的最优贪心策略**。
    实践中常让 ε 从大到小逐渐衰减，兼顾前期探索与后期收敛。

    ## 例子 5 · ε 的探索能力 (Fig 5.8)
    用**固定的 ε-贪心策略**（以最优贪心策略为基底）从 $(s_1,a_1)$ 出发走**一条很长**的轨迹，
    统计它对 125 个 $(s,a)$ 对的访问次数。
    - **ε=1**：完全均匀随机，访问分布很**平**——探索能力最强。
    - **ε=0.5**：探索能力减弱，分布变得**极不均匀**（少数 (s,a) 被反复访问，多数寥寥几次）。

    拖动 **ε** 与**轨迹长度**，看右侧访问次数直方图从“平”变“尖”。
    """)
    return


@app.cell
def _(book_grid, greedy_policy_from_q, np):
    def greedy_base_policy(env):
        P, R = env.build_model()
        g = env.gamma
        v = np.zeros(env.n_states)
        for _ in range(10000):
            q = R + g * P.dot(v)
            vn = q.max(1)
            if np.max(np.abs(vn - v)) < 1e-10:
                break
            v = vn
        return greedy_policy_from_q(R + g * P.dot(v))

    def eps_softened(greedy_pi, eps):
        nS, nA = greedy_pi.shape
        soft = np.full((nS, nA), eps / nA)
        soft[np.arange(nS), greedy_pi.argmax(1)] += 1 - eps
        return soft

    def visit_counts(env, policy, length, seed=0):
        rng = np.random.default_rng(seed)
        counts = np.zeros((env.n_states, env.n_actions))
        env.reset(0)
        a = 0
        for _ in range(length):
            s = env.s_of(*env.agent)
            counts[s, a] += 1
            ns, _, _, _ = env.step(a)
            a = int(rng.choice(env.n_actions, p=policy[ns]))
        return counts

    fig58_base = greedy_base_policy(book_grid())
    return eps_softened, fig58_base, visit_counts


@app.cell
def _(mo):
    eps58_slider = mo.ui.slider(0.1, 1.0, step=0.1, value=1.0,
                                label="探索率 ε（1.0=完全均匀，0.5=偏向贪心）",
                                show_value=True, full_width=True)
    len58_slider = mo.ui.slider(2000, 100000, step=2000, value=50000,
                                label="单条轨迹长度（步）", show_value=True,
                                full_width=True)
    mo.vstack([eps58_slider, len58_slider])
    return eps58_slider, len58_slider


@app.cell
def _(book_grid, eps58_slider, eps_softened, fig58_base, len58_slider,
      np, plt, visit_counts):
    env58 = book_grid()
    pol58 = eps_softened(fig58_base, eps58_slider.value)
    counts58 = visit_counts(env58, pol58, len58_slider.value, seed=0)
    flat = np.sort(counts58.flatten())[::-1]

    fig58, (axb, axh) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axb.bar(np.arange(len(flat)), flat, color="tab:purple")
    axb.set_xlabel("(s,a) 对（按访问次数降序）")
    axb.set_ylabel("访问次数")
    axb.set_title(f"125 个 (s,a) 的访问次数  ε={eps58_slider.value:.1f}")
    axb.grid(True, alpha=0.3)
    axh.imshow(counts58.sum(1).reshape(env58.rows, env58.cols),
               cmap="viridis", origin="upper")
    axh.set_title("每个状态的总访问次数（热力图）")
    axh.set_xticks([]); axh.set_yticks([])
    spread = flat.max() / max(flat[flat > 0].min(), 1)
    fig58.suptitle(f"最高/最低访问次数之比 ≈ {spread:.0f}（ε 越小越不均匀）",
                   fontsize=9)
    fig58.tight_layout()
    fig58
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    把 ε 从 1.0 调到 0.1：左图的直方图从近乎水平（均匀）变得陡峭（极不均），
    右图热力图也从整片均匀变成只有贪心路径附近高亮。
    **结论**：ε 大探索强但策略软（次优），ε 小策略硬但探索弱——这就是探索/利用的权衡。

    ## 小结
    这五个例子把第 5 章的核心直觉都过了一遍：
    1. **大数定律**是 MC 的根基（平均≈期望）；
    2. **MC Basic** 就是“用一条轨迹的回报估 $q$，再贪心改进”；
    3. episode **太短**会让远处状态学不出来；
    4. **ε-greedy** 用软策略保证探索，但 ε 太大会牺牲最优性；
    5. ε 越大**探索越均匀**——这正是要让 ε 先大后小的原因。
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
