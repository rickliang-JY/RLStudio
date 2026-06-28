import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    lang = mo.ui.radio(["中文", "English"], value="中文",
                       label="🌐 语言 / Language", inline=True)
    lang
    return (lang,)


@app.cell(hide_code=True)
def _(lang):
    def t(zh, en):
        return zh if lang.value == "中文" else en

    return (t,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    # 🧪 网格世界自定义实验台 (Custom Grid World Lab)

    一个**完全可自定义**的交互式实验台：自己设网格大小、目标、禁区、各项奖励和
    折扣 $\gamma$，再挑选若干算法**并排对比**它们学到的策略。

    - **上半部分**（实时）：拖动任意滑块/多选，下方"网格预览"和"实时最优策略
      （值迭代）"会**立刻**重画 —— 直接看世界怎么变。
    - **下半部分**（点按钮）：勾选要对比的方法、设好无模型算法的
      $\alpha,\varepsilon,$ episode 数，点「▶ 运行对比」看各方法并排结果。

    | 可选方法 | 类型 | 是否需要模型 |
    |---|---|---|
    | 值迭代 VI / 策略迭代 PI | 动态规划 | ✅ 用 $P,R$ |
    | MC Basic | 无模型 MC（算法5.1，逐 (s,a) 求均值） | ❌ 采样 |
    | MC 探索起点 ES | 无模型 MC（算法5.2，探索起点 + GPI） | ❌ |
    | MC ε-贪心 | 无模型 MC（算法5.3，ε-软策略取代探索起点） | ❌ |
    | Sarsa | 无模型 (on-policy TD) | ❌ |
    | Q-learning | 无模型 (off-policy TD) | ❌ |

    > 约定：进入**目标**即回合结束（episodic）。坐标从 0 开始，$x$ 为列、$y$ 为行。
    > **提示**：把"每步奖励 r_step"设成略小于 0（如 −0.1），可让"尽快到达"成为唯一最优，
    > 各方法的"一致率"更干净；设为 0 即书中经典设定（会有很多并列最优）。
    """, r"""
    # 🧪 Custom Grid World Lab

    A **fully customizable** interactive lab: set the grid size, target, forbidden
    cells, all rewards and the discount $\gamma$ yourself, then pick several
    algorithms to **compare side by side** on the policies they learn.

    - **Top half** (live): drag any slider / multiselect and the "Grid preview" and
      "Live optimal policy (value iteration)" below redraw **instantly** — watch the
      world change directly.
    - **Bottom half** (button): tick the methods to compare, set
      $\alpha,\varepsilon,$ episode count for the model-free algorithms, then click
      "▶ Run comparison" to see each method's result side by side.

    | Method | Type | Needs a model? |
    |---|---|---|
    | Value Iteration VI / Policy Iteration PI | Dynamic programming | ✅ uses $P,R$ |
    | MC Basic | Model-free MC (Algo 5.1, per-$(s,a)$ averaging) | ❌ sampling |
    | MC Exploring Starts ES | Model-free MC (Algo 5.2, exploring starts + GPI) | ❌ |
    | MC ε-greedy | Model-free MC (Algo 5.3, ε-soft policy replaces ES) | ❌ |
    | Sarsa | Model-free (on-policy TD) | ❌ |
    | Q-learning | Model-free (off-policy TD) | ❌ |

    > Convention: entering the **target** ends the episode (episodic). Coordinates
    > start at 0, $x$ is the column, $y$ is the row.
    > **Tip**: setting "per-step reward r_step" slightly below 0 (e.g. −0.1) makes
    > "reach fast" the unique optimum and the "agreement rate" cleaner; setting it to
    > 0 is the classic textbook setup (many tied optima).
    """))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ## 1️⃣ 设计网格：大小 / 目标 / 起点
    """, r"""
    ## 1️⃣ Design the grid: size / target / start
    """))
    return


@app.cell(hide_code=True)
def _(mo, t):
    rows_slider = mo.ui.slider(2, 9, step=1, value=5,
                               label=t("行数", "Rows"), show_value=True)
    cols_slider = mo.ui.slider(2, 9, step=1, value=5,
                               label=t("列数", "Cols"), show_value=True)
    mo.hstack([rows_slider, cols_slider])
    return cols_slider, rows_slider


@app.cell(hide_code=True)
def _(cols_slider, mo, rows_slider, t):
    _C = cols_slider.value
    _R = rows_slider.value
    _cells = [f"{_x},{_y}" for _y in range(_R) for _x in range(_C)]
    _tdef = "2,3" if "2,3" in _cells else _cells[-1]
    _sdef = "0,0" if "0,0" in _cells else _cells[0]
    target_select = mo.ui.dropdown(options=_cells, value=_tdef,
                                   label=t("目标 (x,y)", "Target (x,y)"))
    start_select = mo.ui.dropdown(options=_cells, value=_sdef,
                                  label=t("起点 (x,y)", "Start (x,y)"))
    mo.hstack([target_select, start_select])
    return start_select, target_select


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ### 禁区 forbidden（多选；改网格大小会重置为默认）
    """, r"""
    ### Forbidden cells (multi-select; changing grid size resets to default)
    """))
    return


@app.cell(hide_code=True)
def _(cols_slider, mo, rows_slider, t):
    _C = cols_slider.value
    _R = rows_slider.value
    _opts = [f"{_x},{_y}" for _y in range(_R) for _x in range(_C)]
    _classic = [(1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)]
    _default = [f"{_x},{_y}" for (_x, _y) in _classic if _x < _C and _y < _R]
    forbidden_select = mo.ui.multiselect(
        options=_opts, value=_default,
        label=t("禁区单元格 x,y（坐标从 0 开始）",
                "Forbidden cells x,y (coords start at 0)"),
        full_width=True)
    forbidden_select
    return (forbidden_select,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ## 2️⃣ 设置奖励与折扣
    """, r"""
    ## 2️⃣ Set rewards and discount
    """))
    return


@app.cell(hide_code=True)
def _(mo, t):
    r_t = mo.ui.slider(0.0, 5.0, step=0.5, value=1.0,
                       label=t("到达目标 r_target", "Reach target r_target"),
                       show_value=True)
    r_f = mo.ui.slider(-5.0, 0.0, step=0.5, value=-1.0,
                       label=t("进入禁区 r_forbidden", "Enter forbidden r_forbidden"),
                       show_value=True)
    r_s = mo.ui.slider(-1.0, 0.0, step=0.1, value=-0.1,
                       label=t("每步 r_step", "Per-step r_step"), show_value=True)
    r_b = mo.ui.slider(-5.0, 0.0, step=0.5, value=-1.0,
                       label=t("撞墙 r_boundary", "Hit wall r_boundary"),
                       show_value=True)
    gamma_slider = mo.ui.slider(0.5, 0.99, step=0.01, value=0.9,
                                label=t("折扣 γ", "Discount γ"), show_value=True,
                                full_width=True)
    mo.vstack([mo.hstack([r_t, r_f]), mo.hstack([r_s, r_b]), gamma_slider])
    return gamma_slider, r_b, r_f, r_s, r_t


@app.cell(hide_code=True)
def _(
    GridWorld,
    cols_slider,
    forbidden_select,
    gamma_slider,
    r_b,
    r_f,
    r_s,
    r_t,
    rows_slider,
    start_select,
    target_select,
):
    _C = cols_slider.value
    _R = rows_slider.value
    _ta, _tb = target_select.value.split(",")
    _tx, _ty = int(_ta), int(_tb)
    _sa, _sb = start_select.value.split(",")
    _sx, _sy = int(_sa), int(_sb)
    _forb = []
    for _item in forbidden_select.value:
        _a, _b = _item.split(",")
        _fx, _fy = int(_a), int(_b)
        if _fx < _C and _fy < _R and (_fx, _fy) != (_tx, _ty):
            _forb.append((_fx, _fy))
    env = GridWorld(
        cols=_C, rows=_R, start=(_sx, _sy), target=(_tx, _ty),
        forbidden=tuple(_forb),
        r_target=r_t.value, r_forbidden=r_f.value,
        r_step=r_s.value, r_boundary=r_b.value,
        gamma=gamma_slider.value)
    return (env,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ## 3️⃣ 实时预览（随上面任意滑块即时更新）
    """, r"""
    ## 3️⃣ Live preview (updates instantly with any slider above)
    """))
    return


@app.cell
def _(env, greedy_policy_from_q, mo, t, vi_ep):
    _fig1, _ax1 = env.plot(title=t("网格预览（蓝格=目标，黄格=禁区，蓝星=起点）",
                                   "Grid preview (blue=target, yellow=forbidden, star=start)"))
    _sx, _sy = env.start
    _ax1.plot(_sx, _sy, "*", color="blue", markersize=20)
    _Q = vi_ep(env, env.gamma)
    _pi = greedy_policy_from_q(_Q)
    _fig2 = env.plot_policy(_pi, _Q.max(1),
                            title=t("实时最优策略（值迭代，随滑块更新）",
                                    "Live optimal policy (value iteration)"),
                            precision=1)
    mo.hstack([_fig1, _fig2], justify="center", gap=1.5, widths="equal")
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ## 4️⃣ 方法对比（设好参数后点按钮）

    勾选要并排对比的方法。无模型算法（MC / Sarsa / Q-learning）共用下面的
    $\alpha,\varepsilon,$ episode 数与随机种子；动态规划方法（VI / PI）忽略它们。
    每个面板标题给出该方法贪心策略与**值迭代最优**的"一致率"。
    """, r"""
    ## 4️⃣ Compare methods (set parameters, then click the button)

    Tick the methods to compare side by side. The model-free algorithms
    (MC / Sarsa / Q-learning) share the $\alpha,\varepsilon,$ episode count and
    random seed below; dynamic-programming methods (VI / PI) ignore them. Each
    panel title shows the "agreement rate" between that method's greedy policy and
    the **value-iteration optimum**.
    """))
    return


@app.cell
def _(mo, t):
    # 显示标签随语言切换，但返回值始终是稳定的内部键（供对比 cell 派发）。
    _key_to_label = {
        "值迭代 VI": t("值迭代 VI", "Value Iteration VI"),
        "策略迭代 PI": t("策略迭代 PI", "Policy Iteration PI"),
        "MC Basic": "MC Basic",
        "MC 探索起点 ES": t("MC 探索起点 ES", "MC Exploring Starts ES"),
        "MC ε-贪心": t("MC ε-贪心", "MC ε-greedy"),
        "Sarsa": "Sarsa",
        "Q-learning": "Q-learning",
    }
    _label_to_key = {_v: _k for _k, _v in _key_to_label.items()}
    _defaults = ["值迭代 VI", "MC Basic", "MC 探索起点 ES", "MC ε-贪心"]
    method_select = mo.ui.multiselect(
        options=_label_to_key,
        value=[_key_to_label[_k] for _k in _defaults],
        label=t("选择要对比的方法（多选）", "Methods to compare (multi-select)"),
        full_width=True)
    alpha_s = mo.ui.slider(0.05, 0.9, step=0.05, value=0.1,
                           label=t("学习率 α（无模型）", "Learning rate α (model-free)"),
                           show_value=True, full_width=True)
    eps_s = mo.ui.slider(0.0, 0.5, step=0.05, value=0.1,
                         label=t("探索率 ε（无模型）", "Exploration ε (model-free)"),
                         show_value=True, full_width=True)
    ep_s = mo.ui.slider(1000, 40000, step=1000, value=5000,
                        label=t("训练 episode 数（无模型）",
                                "Episode count (model-free)"),
                        show_value=True, full_width=True)
    seed_s = mo.ui.slider(0, 9, step=1, value=0,
                          label=t("随机种子", "Random seed"), show_value=True,
                          full_width=True)
    run_btn = mo.ui.run_button(label=t("▶ 运行对比", "▶ Run comparison"))
    mo.vstack([method_select, alpha_s, eps_s, ep_s, seed_s, run_btn])
    return alpha_s, ep_s, eps_s, method_select, run_btn, seed_s


@app.cell
def _(
    alpha_s,
    env,
    ep_s,
    eps_s,
    greedy_policy_from_q,
    mc_basic,
    mc_epsilon,
    mc_exploring_starts,
    method_select,
    mo,
    pi_ep,
    plt,
    q_learning,
    run_btn,
    sarsa,
    seed_s,
    t,
    vi_ep,
):
    mo.stop(not run_btn.value,
            mo.md(t("⬆️ **选好方法和参数后，点击「▶ 运行对比」**，这里显示并排结果。",
                    "⬆️ **Pick methods and parameters, then click「▶ Run comparison」**"
                    " — results appear here side by side.")))
    mo.stop(len(method_select.value) == 0,
            mo.md(t("请至少选择一种方法。", "Please select at least one method.")))

    _max_steps = max(4 * env.n_states, 60)
    _ref_pi = greedy_policy_from_q(vi_ep(env, env.gamma))
    _disp = {
        "值迭代 VI": t("值迭代 VI", "Value Iteration VI"),
        "策略迭代 PI": t("策略迭代 PI", "Policy Iteration PI"),
        "MC 探索起点 ES": t("MC 探索起点 ES", "MC Exploring Starts ES"),
        "MC ε-贪心": t("MC ε-贪心", "MC ε-greedy"),
    }
    _model_free = {
        "MC Basic": mc_basic,
        "MC 探索起点 ES": mc_exploring_starts,
        "MC ε-贪心": mc_epsilon,
        "Sarsa": sarsa,
        "Q-learning": q_learning,
    }

    _res = []
    for _name in method_select.value:
        if _name == "值迭代 VI":
            _Q = vi_ep(env, env.gamma)
        elif _name == "策略迭代 PI":
            _Q = pi_ep(env, env.gamma)
        else:
            _Q = _model_free[_name](env, env.gamma, alpha_s.value, eps_s.value,
                                    ep_s.value, seed_s.value, _max_steps)
        _pi = greedy_policy_from_q(_Q)
        _agree = float((_pi.argmax(1) == _ref_pi.argmax(1)).mean())
        _res.append((_name, _pi, _Q.max(1), _agree))

    _n = len(_res)
    _fig, _axes = plt.subplots(1, _n, figsize=(4.2 * _n, 4.6), squeeze=False)
    for _i, (_name, _pi, _v, _agree) in enumerate(_res):
        _ax = _axes[0][_i]
        env._base_ax(
            _ax,
            f"{_disp.get(_name, _name)}  ({t('与VI一致率', 'agree w/ VI')} {_agree:.0%})")
        env._draw_policy(_ax, _pi)
        for _s in range(env.n_states):
            _x, _y = env.xy_of(_s)
            _ax.text(_x, _y + 0.34, f"{_v[_s]:.1f}", ha="center", va="center",
                     fontsize=6, color="black")
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ## 玩法建议
    - **看三种 MC 的差异**：
      - **MC Basic** 每轮把每个 $(s,a)$ 都重新评估、再贪心，单位预算下最准，
        但需要能从任意 $(s,a)$ 重置（生成式模型假设）。
      - **MC 探索起点 ES** 靠随机起点覆盖，样本利用率低，**需要很多 episode**
        才追上（把 episode 数拉到几万看一致率怎么涨）。
      - **MC ε-贪心** 用 ε 软策略取代探索起点，**固定从起点出发**，覆盖好坏取决于
        $\varepsilon$：ε 太小远处状态几乎访问不到（一致率低），调大 ε 明显改善。
    - **看探索的作用**：把 $\varepsilon$ 调到 0，Sarsa/Q-learning/MC ε 常学不全 →
      调大 $\varepsilon$ 或 episode 数再看一致率怎么涨。
    - **看奖励塑形**：把 r_forbidden 调得很负（如 −5），禁区附近策略会绕开；
      把 r_step 调成 0，会出现大量并列最优，"一致率"随之波动。
    - **看模型 vs 无模型 vs 自举**：VI/PI 一步到位（需要 $P,R$）；MC 只靠完整回报、
      慢；Sarsa/Q 用 TD 自举，采样效率通常高于 MC。
    """, r"""
    ## Tips for playing
    - **See the difference between the three MCs**:
      - **MC Basic** re-evaluates every $(s,a)$ each round then goes greedy — most
        accurate per unit budget, but needs to reset from any $(s,a)$ (a generative
        model assumption).
      - **MC Exploring Starts ES** relies on random starts for coverage; it is
        sample-inefficient and **needs many episodes** to catch up (push the episode
        count into the tens of thousands and watch the agreement rate climb).
      - **MC ε-greedy** replaces exploring starts with an ε-soft policy and **always
        starts from the start cell**; coverage depends on $\varepsilon$ — too small
        and far states are barely visited (low agreement); raising ε clearly helps.
    - **See the role of exploration**: set $\varepsilon$ to 0 and Sarsa/Q-learning/MC
      ε often fail to learn fully → raise $\varepsilon$ or the episode count and watch
      the agreement rate rise.
    - **See reward shaping**: make r_forbidden very negative (e.g. −5) and policies
      detour around forbidden cells; set r_step to 0 and many tied optima appear,
      making the "agreement rate" fluctuate.
    - **See model vs model-free vs bootstrapping**: VI/PI are one-shot (need $P,R$);
      MC relies only on full returns and is slow; Sarsa/Q bootstrap with TD and are
      usually more sample-efficient than MC.
    """))
    return


@app.cell
def _(np):
    # ---- 五种方法的实现（全部按 episodic：进入目标即终止）----
    def _eps_greedy(_qrow, _eps, _nA, _rng):
        if _rng.random() < _eps:
            return int(_rng.integers(_nA))
        return int(np.argmax(_qrow))

    def vi_ep(_env, _gamma, tol=1e-10):
        _P, _R = _env.build_model()
        _tgt = _env.s_of(*_env.target)
        _v = np.zeros(_env.n_states)
        while True:
            _Q = _R + _gamma * _P.dot(_v)
            _vn = _Q.max(1)
            _vn[_tgt] = 0.0
            if np.max(np.abs(_vn - _v)) < tol:
                return _R + _gamma * _P.dot(_vn)
            _v = _vn

    def pi_ep(_env, _gamma, max_iter=100):
        _P, _R = _env.build_model()
        _tgt = _env.s_of(*_env.target)
        _n, _nA = _env.n_states, _env.n_actions
        _pol = np.zeros(_n, dtype=int)
        _I = np.eye(_n)
        _Ppi = np.zeros((_n, _n))
        _rpi = np.zeros(_n)
        for _ in range(max_iter):
            _Ppi[:] = 0.0
            for _s in range(_n):
                _a = _pol[_s]
                _ns = int(_P[_s, _a].argmax())
                _rpi[_s] = _R[_s, _a]
                if _ns != _tgt:
                    _Ppi[_s, _ns] = _gamma
            _v = np.linalg.solve(_I - _Ppi, _rpi)
            _v[_tgt] = 0.0
            _Q = _R + _gamma * _P.dot(_v)
            _newpol = _Q.argmax(1)
            if np.array_equal(_newpol, _pol):
                return _Q
            _pol = _newpol
        return _R + _gamma * _P.dot(np.linalg.solve(_I - _Ppi, _rpi))

    def mc_basic(_env, _gamma, _alpha, _eps, _episodes, _seed, _max_steps):
        # 算法 5.1：最朴素的 MC。每轮对“每个 (s,a)”各采若干条 episode、
        # 直接求平均得到 q，再贪心改进策略。确定性策略、无 ε、无自举。
        _nS, _nA = _env.n_states, _env.n_actions
        _tgt = _env.s_of(*_env.target)
        _pol = np.zeros(_nS, dtype=int)
        _Q = np.zeros((_nS, _nA))
        _n_iters = 10
        _per = max(1, _episodes // (_n_iters * _nS * _nA))   # 每个(s,a)采多少条
        for _it in range(_n_iters):
            for _s in range(_nS):
                if _s == _tgt:
                    continue
                for _a in range(_nA):
                    _tot = 0.0
                    for _k in range(_per):
                        _env.reset(_s)
                        _act = _a
                        _G, _disc = 0.0, 1.0
                        for _t in range(_max_steps):
                            _ns, _r, _done, _ = _env.step(_act)
                            _G += _disc * _r
                            _disc *= _gamma
                            if _done:
                                break
                            _act = _pol[_ns]
                        _tot += _G
                    _Q[_s, _a] = _tot / _per
            _pol = _Q.argmax(1)
        return _Q

    def mc_exploring_starts(_env, _gamma, _alpha, _eps, _episodes, _seed, _max_steps):
        # 算法 5.2：广义策略迭代 + 探索起点。每条 episode 从随机 (s0,a0) 出发、
        # 之后跟随确定性贪心策略；用样本均值更新 Q，并即时贪心改进。
        _rng = np.random.default_rng(_seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _Q = np.zeros((_nS, _nA))
        _N = np.zeros((_nS, _nA))
        _pol = np.zeros(_nS, dtype=int)
        for _ep in range(_episodes):
            _cs = int(_rng.integers(_nS))         # 探索起点：随机状态
            _a = int(_rng.integers(_nA))          # 随机首动作
            _env.reset(_cs)
            _traj = []
            for _t in range(_max_steps):
                _ns, _r, _done, _ = _env.step(_a)
                _traj.append((_cs, _a, _r))
                _cs = _ns
                if _done:
                    break
                _a = _pol[_cs]
            _G = 0.0
            for (_s, _a, _r) in reversed(_traj):
                _G = _r + _gamma * _G
                _N[_s, _a] += 1
                _Q[_s, _a] += (_G - _Q[_s, _a]) / _N[_s, _a]
                _pol[_s] = int(np.argmax(_Q[_s]))
        return _Q

    def mc_epsilon(_env, _gamma, _alpha, _eps, _episodes, _seed, _max_steps):
        # 算法 5.3：用 ε-贪心软策略取代探索起点。固定从起点出发，靠 ε 保证探索；
        # 覆盖好坏取决于 ε（ε 太小远处状态难被访问）。样本均值更新 Q。
        _rng = np.random.default_rng(_seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _Q = np.zeros((_nS, _nA))
        _N = np.zeros((_nS, _nA))
        for _ep in range(_episodes):
            _cs = _env.reset()                    # 固定起点，无探索起点
            _traj = []
            for _t in range(_max_steps):
                _a = _eps_greedy(_Q[_cs], _eps, _nA, _rng)
                _ns, _r, _done, _ = _env.step(_a)
                _traj.append((_cs, _a, _r))
                _cs = _ns
                if _done:
                    break
            _G = 0.0
            for (_s, _a, _r) in reversed(_traj):
                _G = _r + _gamma * _G
                _N[_s, _a] += 1
                _Q[_s, _a] += (_G - _Q[_s, _a]) / _N[_s, _a]
        return _Q

    def sarsa(_env, _gamma, _alpha, _eps, _episodes, _seed, _max_steps):
        _rng = np.random.default_rng(_seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _Q = np.zeros((_nS, _nA))
        for _ep in range(_episodes):
            _s = _env.reset(int(_rng.integers(_nS)))
            _a = _eps_greedy(_Q[_s], _eps, _nA, _rng)
            for _t in range(_max_steps):
                _ns, _r, _done, _ = _env.step(_a)
                _na = _eps_greedy(_Q[_ns], _eps, _nA, _rng)
                _target = _r + (0.0 if _done else _gamma * _Q[_ns, _na])
                _Q[_s, _a] += _alpha * (_target - _Q[_s, _a])
                _s, _a = _ns, _na
                if _done:
                    break
        return _Q

    def q_learning(_env, _gamma, _alpha, _eps, _episodes, _seed, _max_steps):
        _rng = np.random.default_rng(_seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _Q = np.zeros((_nS, _nA))
        for _ep in range(_episodes):
            _s = _env.reset(int(_rng.integers(_nS)))
            for _t in range(_max_steps):
                _a = _eps_greedy(_Q[_s], _eps, _nA, _rng)
                _ns, _r, _done, _ = _env.step(_a)
                _target = _r + (0.0 if _done else _gamma * _Q[_ns].max())
                _Q[_s, _a] += _alpha * (_target - _Q[_s, _a])
                _s = _ns
                if _done:
                    break
        return _Q

    return (
        mc_basic,
        mc_epsilon,
        mc_exploring_starts,
        pi_ep,
        q_learning,
        sarsa,
        vi_ep,
    )


@app.cell
def _():
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from gridworld import GridWorld, greedy_policy_from_q
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return GridWorld, greedy_policy_from_q, np, plt


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
