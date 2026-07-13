# /// script
# requires-python = ">=3.10"
# dependencies = [
#     # Pinned to the set this lab is verified against. Left unpinned, molab/uv installs
#     # the newest marimo + anywidget, whose ESM loader fails to load Wanderland's 3D
#     # widget bundle ("missing a default export" — the bundle IS AFM-compliant; the
#     # newer loader just can't load it). These versions render the 3D scene correctly.
#     "marimo==0.23.10",
#     "anywidget==0.11.0",
#     "wanderland==0.1.1",
#     "numpy",
#     "matplotlib",
# ]
# ///

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
    # 🌱 Wanderland 网格世界实验台

    和隔壁 `custom_gridworld_lab.py` **同一套世界、同一批算法**，但不再用 matplotlib
    的箭头表格 —— 而是让 **Mo 苔藓球在低多边形 3D 世界里,把学到的最优策略一步步走出来**。

    - **上半**（实时）：拖滑块设计世界（大小 / 目标 / 起点 / 禁区 / 奖励 / $\gamma$），
      下方 2D 预览立刻重画。
    - **下半**（点按钮）：选一个算法求解，点「▶ 求解并让 Mo 行走」→ 3D 场景出现,
      按场景里的 **▶ Run My Code** 看 Mo 走完这条最优路径。可拖动鼠标转动镜头。

    > **禁区 = 熔岩 🌋**：Wanderland 里禁区渲染成"可走但致命"的熔岩。最优策略会
    > **优雅地绕开**它 —— 这正是好看的地方。反过来,一个训练不足的无模型策略可能
    > **一头走进熔岩**（Mo 会掉进去），这恰好把"策略没学好"可视化了出来。
    >
    > 约定同隔壁：进入**目标**即回合结束;坐标从 0 开始,$x$ 为列、$y$ 为行。
    """, r"""
    # 🌱 Wanderland Grid World Lab

    The **same worlds and the same algorithms** as the sibling
    `custom_gridworld_lab.py`, but instead of matplotlib arrow-tables, watch **Mo the
    Mossball walk the learned optimal policy step by step in a low-poly 3D world**.

    - **Top** (live): drag sliders to design the world (size / target / start /
      forbidden / rewards / $\gamma$); the 2D preview below redraws instantly.
    - **Bottom** (button): pick an algorithm, click "▶ Solve & let Mo walk" → the 3D
      scene appears; press the in-scene **▶ Run My Code** to watch Mo walk the optimal
      path. Drag to orbit the camera.

    > **Forbidden = lava 🌋**: forbidden cells render as "walkable but deadly" lava. The
    > optimal policy **gracefully routes around** it — that's the charming part.
    > Conversely, an under-trained model-free policy may **walk straight into the lava**
    > (Mo falls in), which nicely visualizes "the policy hasn't converged".
    >
    > Same convention as the sibling: entering the **target** ends the episode;
    > coordinates start at 0, $x$ is the column, $y$ is the row.
    """))
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""## 1️⃣ 设计网格：大小 / 目标 / 起点""",
            r"""## 1️⃣ Design the grid: size / target / start"""))
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
    mo.md(t(r"""### 禁区 forbidden（多选;改网格大小会重置为默认;渲染成熔岩 🌋）""",
            r"""### Forbidden cells (multi-select; resets on size change; rendered as lava 🌋)"""))
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
    mo.md(t(r"""### 🧱 障碍:墙 / 迷宫(墙彻底挡路,区别于"可踩但扣分"的熔岩)""",
            r"""### 🧱 Obstacles: walls / maze (walls fully block — unlike walkable, penalized lava)"""))
    return


@app.cell(hide_code=True)
def _(mo, t):
    obstacle_mode = mo.ui.radio(
        options={t("手动摆墙", "Place by hand"): "manual",
                 t("随机迷宫 🎲", "Random maze 🎲"): "random"},
        value=t("手动摆墙", "Place by hand"), inline=True,
        label=t("墙的来源", "Walls from"))
    obstacle_skin = mo.ui.dropdown(
        options={t("墙块 🧱", "Wall blocks 🧱"): "walls",
                 t("水沟 💧", "Water moat 💧"): "water"},
        value=t("墙块 🧱", "Wall blocks 🧱"),
        label=t("障碍外观", "Obstacle look"))
    mo.hstack([obstacle_mode, obstacle_skin])
    return obstacle_mode, obstacle_skin


@app.cell(hide_code=True)
def _(cols_slider, mo, rows_slider, t):
    _opts = [f"{_x},{_y}" for _y in range(rows_slider.value)
             for _x in range(cols_slider.value)]
    walls_select = mo.ui.multiselect(
        options=_opts, value=[],
        label=t("手动墙格 x,y(仅手动模式生效)",
                "Manual wall cells x,y (manual mode only)"),
        full_width=True)
    maze_density = mo.ui.slider(0.1, 0.4, step=0.05, value=0.28,
                                label=t("迷宫墙密度(随机模式)", "Maze density (random mode)"),
                                show_value=True)
    maze_seed = mo.ui.slider(0, 30, step=1, value=0,
                             label=t("迷宫随机种子 🎲", "Maze seed 🎲"), show_value=True)
    maze_lava = mo.ui.slider(0, 8, step=1, value=3,
                             label=t("迷宫熔岩数(随机模式,撒在最短路之外)",
                                     "Maze lava count (random mode, off the shortest path)"),
                             show_value=True)
    mo.vstack([walls_select, mo.hstack([maze_density, maze_seed, maze_lava])])
    return maze_density, maze_lava, maze_seed, walls_select


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""## 2️⃣ 设置奖励与折扣""", r"""## 2️⃣ Set rewards and discount"""))
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
    MazeGridWorld,
    cols_slider,
    forbidden_select,
    gamma_slider,
    generate_maze,
    maze_density,
    maze_lava,
    maze_seed,
    obstacle_mode,
    r_b,
    r_f,
    r_s,
    r_t,
    random_lava,
    rows_slider,
    start_select,
    target_select,
    walls_select,
):
    _C = cols_slider.value
    _R = rows_slider.value
    _tx, _ty = (int(_v) for _v in target_select.value.split(","))
    _sx, _sy = (int(_v) for _v in start_select.value.split(","))

    if obstacle_mode.value == "random":
        _walls = list(generate_maze(_C, _R, (_sx, _sy), (_tx, _ty),
                                    maze_seed.value, maze_density.value))
    else:
        _walls = []
        for _item in walls_select.value:
            _wx, _wy = (int(_v) for _v in _item.split(","))
            if _wx < _C and _wy < _R and (_wx, _wy) not in ((_sx, _sy), (_tx, _ty)):
                _walls.append((_wx, _wy))
    _wset = set(_walls)

    # Lava: random mode scatters it OFF a shortest path (so Mo always has a clean
    # lava-free route); manual mode keeps whatever lava the user selected.
    if obstacle_mode.value == "random":
        _forb = list(random_lava(_C, _R, (_sx, _sy), (_tx, _ty),
                                 _walls, maze_lava.value, maze_seed.value))
    else:
        _forb = []
        for _item in forbidden_select.value:
            _fx, _fy = (int(_v) for _v in _item.split(","))
            if (_fx < _C and _fy < _R and (_fx, _fy) != (_tx, _ty)
                    and (_fx, _fy) not in _wset):
                _forb.append((_fx, _fy))

    env = MazeGridWorld(
        cols=_C, rows=_R, start=(_sx, _sy), target=(_tx, _ty),
        walls=tuple(_walls), forbidden=tuple(_forb),
        r_target=r_t.value, r_forbidden=r_f.value,
        r_step=r_s.value, r_boundary=r_b.value,
        gamma=gamma_slider.value)
    return (env,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""## 3️⃣ 世界预览（随上面任意滑块即时更新）""",
            r"""## 3️⃣ World preview (updates instantly with any slider above)"""))
    return


@app.cell
def _(env, mo, patches, t):
    _fig, _ax = env.plot(title=t("蓝格=目标  黄格=熔岩  灰格=墙  蓝星=起点",
                                 "blue=target  yellow=lava  grey=wall  star=start"))
    for (_wx, _wy) in env.wall_set:
        _ax.add_patch(patches.Rectangle((_wx - 0.5, _wy - 0.5), 1, 1,
                                        facecolor=(0.5, 0.53, 0.57)))
    _sx, _sy = env.start
    _ax.plot(_sx, _sy, "*", color="blue", markersize=20)
    mo.hstack([_fig], justify="center")
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ## 4️⃣ 选算法,让 Mo 行走

    选一个算法求解出最优策略。动态规划（VI / PI）用模型 $P,R$,秒出;无模型算法
    （MC / Sarsa / Q-learning）用下面的 $\alpha,\varepsilon$、episode 数、随机种子采样学习。
    设好后点「▶ 求解并让 Mo 行走」。
    """, r"""
    ## 4️⃣ Pick an algorithm, let Mo walk

    Pick one algorithm to solve for the optimal policy. Dynamic programming (VI / PI)
    uses the model $P,R$ and is instant; model-free (MC / Sarsa / Q-learning) samples
    with the $\alpha,\varepsilon$, episode count and seed below. Set them, then click
    "▶ Solve & let Mo walk".
    """))
    return


@app.cell(hide_code=True)
def _(mo, t):
    _key_to_label = {
        "值迭代 VI": t("值迭代 VI", "Value Iteration VI"),
        "策略迭代 PI": t("策略迭代 PI", "Policy Iteration PI"),
        "MC Basic": "MC Basic",
        "MC 探索起点 ES": t("MC 探索起点 ES", "MC Exploring Starts ES"),
        "MC ε-贪心": t("MC ε-贪心", "MC ε-greedy"),
        "Sarsa": "Sarsa",
        "Q-learning": "Q-learning",
    }
    algo_select = mo.ui.dropdown(
        options={_v: _k for _k, _v in _key_to_label.items()},
        value=_key_to_label["值迭代 VI"],
        label=t("算法", "Algorithm"))
    char_dd = mo.ui.dropdown(
        options={"Mo": "mo", "Rover": "rover"},
        value="Mo", label=t("角色", "Character"))
    speed_s = mo.ui.slider(0.5, 3.0, step=0.5, value=1.0,
                           label=t("动画速度", "Animation speed"), show_value=True)
    alpha_s = mo.ui.slider(0.05, 0.9, step=0.05, value=0.1,
                           label=t("学习率 α（无模型）", "Learning rate α (model-free)"),
                           show_value=True, full_width=True)
    eps_s = mo.ui.slider(0.0, 0.5, step=0.05, value=0.1,
                         label=t("探索率 ε（无模型）", "Exploration ε (model-free)"),
                         show_value=True, full_width=True)
    ep_s = mo.ui.slider(1000, 40000, step=1000, value=5000,
                        label=t("训练 episode 数（无模型）", "Episode count (model-free)"),
                        show_value=True, full_width=True)
    seed_s = mo.ui.slider(0, 9, step=1, value=0,
                          label=t("随机种子", "Random seed"), show_value=True)
    run_btn = mo.ui.run_button(label=t("▶ 求解并让 Mo 行走", "▶ Solve & let Mo walk"))
    mo.vstack([
        mo.hstack([algo_select, char_dd, speed_s]),
        alpha_s, eps_s, ep_s,
        mo.hstack([seed_s]),
        run_btn,
    ])
    return algo_select, alpha_s, char_dd, ep_s, eps_s, run_btn, seed_s, speed_s


@app.cell(hide_code=True)
def _(
    Puzzle,
    algo_select,
    alpha_s,
    build_puzzle,
    env,
    ep_s,
    eps_s,
    mc_basic,
    mc_epsilon,
    mc_exploring_starts,
    mo,
    obstacle_skin,
    pi_ep,
    q_learning,
    rollout_greedy,
    run_btn,
    sarsa,
    seed_s,
    t,
    traj_to_commands,
    vi_ep,
):
    # Solve, then hand the next cell a *plan* (puzzle + command list) rather than a World.
    # Building the World lives in its own cell on purpose — see the note there.
    if not run_btn.value:
        plan = None
        _hint = mo.md(t("⬆️ **设好世界和算法后,点「▶ 求解并让 Mo 行走」**,这里出现 3D 场景。",
                        "⬆️ **Design the world & pick an algorithm, then click "
                        "「▶ Solve & let Mo walk」** — the 3D scene appears here."))
    else:
        _max_steps = max(4 * env.n_states, 60)
        _algo = algo_select.value
        _model_free = {
            "MC Basic": mc_basic,
            "MC 探索起点 ES": mc_exploring_starts,
            "MC ε-贪心": mc_epsilon,
            "Sarsa": sarsa,
            "Q-learning": q_learning,
        }
        if _algo == "值迭代 VI":
            _Q = vi_ep(env, env.gamma)
        elif _algo == "策略迭代 PI":
            _Q = pi_ep(env, env.gamma)
        else:
            _Q = _model_free[_algo](env, env.gamma, alpha_s.value, eps_s.value,
                                    ep_s.value, seed_s.value, _max_steps)

        _states, _acts = rollout_greedy(env, _Q, _max_steps)
        _init_h, _names = traj_to_commands(env, _acts)
        plan = {
            "algo": _algo,
            "puzzle": build_puzzle(env, Puzzle, _init_h, obstacle_skin.value),
            "names": _names,
        }
        _hint = None
    _hint
    return (plan,)


@app.cell(hide_code=True)
def _(World, char_dd, mo, plan, speed_s):
    # ONE World per plan, built here and nowhere else.
    #
    # When playback finishes, Wanderland's frontend writes the `state` trait — a value
    # change on `world` — so every cell that *reads* `world` re-runs. Wanderland is built
    # for that: `act(play=False)`/`load()` are idempotent, and re-loading an unchanged
    # timeline is a no-op that leaves Mo standing on the goal. But the guard compares
    # against `self.timeline`, so it only works if the *same* World survives the re-run.
    # Rebuild the World inside the driving cell and every finished run mints a fresh
    # (empty-timeline) widget — which is exactly Mo snapping back to the start.
    #
    # This cell doesn't read `world`, so playback never re-runs it; the World is rebuilt
    # only when the plan / character / speed actually change.
    world = (
        mo.ui.anywidget(World(plan["puzzle"], character=char_dd.value, speed=speed_s.value))
        if plan
        else None
    )
    return (world,)


@app.cell(hide_code=True)
def _(mo, plan, t, world):
    mo.stop(world is None)

    _res = world.act(plan["names"], play=False)  # idempotent: a no-op on playback re-runs

    if _res.get("died"):
        _verdict = t("🌋 Mo 走进了熔岩(踩到禁区)—— 这个策略没学好,试试换算法/加 episode/调奖励。",
                     "🌋 Mo walked into the lava (stepped on a forbidden cell) — this "
                     "policy isn't good; try another algorithm / more episodes / reshape rewards.")
    elif _res.get("reached_goal"):
        _verdict = t(f"✅ Mo 沿最优策略到达目标,共 {_res.get('steps', 0)} 步。按场景里的 ▶ Run 观看。",
                     f"✅ Mo reached the goal along the policy in {_res.get('steps', 0)} steps. "
                     "Press ▶ Run in the scene to watch.")
    else:
        _verdict = t(f"⚠️ {_res.get('steps', 0)} 步内没到目标(策略可能在打转)。",
                     f"⚠️ Did not reach the goal within {_res.get('steps', 0)} steps "
                     "(the policy may be looping).")

    mo.vstack([
        mo.md(t(f"### 🎬 {plan['algo']} 学到的策略 —— 让 Mo 走一遍",
                f"### 🎬 The policy learned by {plan['algo']} — watch Mo walk it")),
        world,
        mo.md(_verdict),
    ])
    return


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ## 5️⃣ 看 Mo 学习的过程 🎓

    同一个算法,这次**记录训练途中的策略快照**,从"没怎么练"到"练到最优"排成一排。
    点按钮训练并记录,再拖「训练进度」滑块 —— 看 Mo 从**一头栽进熔岩 / 原地打转**,
    一步步学成**直奔目标**。(无模型算法按 episode 预算阶梯记录;VI/PI 按每轮迭代记录。)
    """, r"""
    ## 5️⃣ Watch Mo learn 🎓

    Same algorithm, but this time we **record policy snapshots during training**, lined
    up from "barely trained" to "fully solved". Click to train & record, then drag the
    "Training progress" slider — watch Mo go from **falling into lava / looping in place**
    to **heading straight for the goal**. (Model-free: an episode-budget ladder; VI/PI:
    one snapshot per iteration.)
    """))
    return


@app.cell
def _(mo, t):
    learn_btn = mo.ui.run_button(
        label=t("▶ 训练并记录学习过程", "▶ Train & record the learning process"))
    learn_btn
    return (learn_btn,)


@app.cell
def _(
    algo_select,
    alpha_s,
    env,
    ep_s,
    eps_s,
    learn_btn,
    learning_snapshots,
    mo,
    seed_s,
    t,
):
    if learn_btn.value:
        _max_steps = max(4 * env.n_states, 60)
        snapshots = learning_snapshots(algo_select.value, env, alpha_s.value,
                                       eps_s.value, ep_s.value, seed_s.value, _max_steps)
        _msg = mo.md(t(f"已记录 **{len(snapshots)}** 个学习快照。拖下面的「训练进度」看 Mo 逐步学会。",
                       f"Recorded **{len(snapshots)}** snapshots. Drag "
                       "\"Training progress\" below to watch Mo learn."))
    else:
        snapshots = []
        _msg = mo.md(t("⬆️ 点「▶ 训练并记录学习过程」开始。",
                       "⬆️ Click \"▶ Train & record the learning process\" to begin."))
    _msg
    return (snapshots,)


@app.cell
def _(mo, snapshots, t):
    stage = mo.ui.slider(0, max(len(snapshots) - 1, 0), step=1,
                         value=max(len(snapshots) - 1, 0),
                         label=t("训练进度", "Training progress"),
                         show_value=True, full_width=True)
    stage if snapshots else None
    return (stage,)


@app.cell(hide_code=True)
def _(
    Puzzle,
    build_puzzle,
    env,
    mo,
    obstacle_skin,
    rollout_greedy,
    snapshots,
    stage,
    t,
    traj_to_commands,
):
    # Same three-cell split as section 4: plan here, World next door, drive after that.
    if not snapshots:
        learn_plan = None
        _hint = mo.md(t("(记录学习过程后,这里出现可拖动回放的 3D 场景。)",
                        "(After recording, a scrubbable 3D replay appears here.)"))
    else:
        _k = min(stage.value, len(snapshots) - 1)
        _label, _Q = snapshots[_k]
        _max_steps = max(4 * env.n_states, 60)
        _states, _acts = rollout_greedy(env, _Q, _max_steps)
        _init_h, _names = traj_to_commands(env, _acts)
        learn_plan = {
            "puzzle": build_puzzle(env, Puzzle, _init_h, obstacle_skin.value),
            "names": _names,
            "label": _label,
            "k": _k,
            "n": len(snapshots),
        }
        _hint = None
    _hint
    return (learn_plan,)


@app.cell(hide_code=True)
def _(World, char_dd, learn_plan, mo, speed_s):
    # Persistent World for the learning replay — same reasoning as the walk cell above:
    # it must survive the re-run that a finished playback triggers, or Mo resets.
    # Scrubbing the "training progress" slider changes learn_plan, so a new stage does
    # get a fresh World — which is what we want.
    world_learn = (
        mo.ui.anywidget(World(learn_plan["puzzle"], character=char_dd.value,
                              speed=speed_s.value))
        if learn_plan
        else None
    )
    return (world_learn,)


@app.cell(hide_code=True)
def _(learn_plan, mo, t, world_learn):
    mo.stop(world_learn is None)

    _res = world_learn.act(learn_plan["names"], play=False)  # idempotent on re-runs

    if _res.get("died"):
        _v = t("🌋 走进熔岩", "🌋 into the lava")
    elif _res.get("reached_goal"):
        _v = t(f"✅ 到达目标,{_res.get('steps', 0)} 步",
               f"✅ reached the goal in {_res.get('steps', 0)} steps")
    else:
        _v = t(f"⚠️ {_res.get('steps', 0)} 步未达(打转)",
               f"⚠️ not reached in {_res.get('steps', 0)} steps (looping)")

    mo.vstack([
        mo.md(t(f"### 🎓 学习阶段 {learn_plan['k'] + 1}/{learn_plan['n']} · "
                f"{learn_plan['label']} — {_v}",
                f"### 🎓 Stage {learn_plan['k'] + 1}/{learn_plan['n']} · "
                f"{learn_plan['label']} — {_v}")),
        world_learn,
        mo.md(t("拖上面的「训练进度」对比不同训练量下 Mo 的走法;按场景内 ▶ Run 播放。",
                "Drag \"Training progress\" above to compare Mo's walk at different amounts "
                "of training; press ▶ Run in the scene to play.")),
    ])
    return


@app.cell
def _(mc_basic, mc_epsilon, mc_exploring_starts, np, q_learning, sarsa):
    # ---- learning trace: a ladder of policy snapshots from "clueless" to "solved" ----
    def _traced_vi(_env, _gamma, _max_sweeps=15, _tol=1e-9):
        _P, _R = _env.build_model()
        _tgt = _env.s_of(*_env.target)
        _v = np.zeros(_env.n_states)
        _snaps = []
        for _k in range(1, _max_sweeps + 1):
            _Q = _R + _gamma * _P.dot(_v)
            _vn = _Q.max(1)
            _vn[_tgt] = 0.0
            _snaps.append((f"sweep {_k}", _R + _gamma * _P.dot(_vn)))
            if np.max(np.abs(_vn - _v)) < _tol:
                break
            _v = _vn
        return _snaps

    def _traced_pi(_env, _gamma, _max_iter=12):
        _P, _R = _env.build_model()
        _tgt = _env.s_of(*_env.target)
        _n, _nA = _env.n_states, _env.n_actions
        _pol = np.zeros(_n, dtype=int)
        _I = np.eye(_n)
        _Ppi = np.zeros((_n, _n))
        _rpi = np.zeros(_n)
        _snaps = []
        for _k in range(1, _max_iter + 1):
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
            _snaps.append((f"iter {_k}", _Q))
            _newpol = _Q.argmax(1)
            if np.array_equal(_newpol, _pol):
                break
            _pol = _newpol
        return _snaps

    _MODEL_FREE_TRACE = {
        "MC Basic": mc_basic,
        "MC 探索起点 ES": mc_exploring_starts,
        "MC ε-贪心": mc_epsilon,
        "Sarsa": sarsa,
        "Q-learning": q_learning,
    }

    def learning_snapshots(_algo, _env, _alpha, _eps, _episodes, _seed, _max_steps):
        """A list of (label, Q) from little training to full — the learning film strip.

        Model-free: re-train from scratch at a ladder of episode budgets (each a valid
        "policy after N episodes" checkpoint, seed fixed). VI/PI: one snapshot per sweep
        / iteration.
        """
        if _algo == "值迭代 VI":
            return _traced_vi(_env, _env.gamma)
        if _algo == "策略迭代 PI":
            return _traced_pi(_env, _env.gamma)
        _fn = _MODEL_FREE_TRACE[_algo]
        _snaps, _seen = [], set()
        for _frac in (0.002, 0.01, 0.04, 0.1, 0.25, 0.5, 1.0):
            _n = max(1, int(round(_frac * _episodes)))
            if _n in _seen:
                continue
            _seen.add(_n)
            _Q = _fn(_env, _env.gamma, _alpha, _eps, _n, _seed, _max_steps)
            _snaps.append((f"{_n} ep", _Q))
        return _snaps

    return (learning_snapshots,)


@app.cell(hide_code=True)
def _(mo, t):
    mo.md(t(r"""
    ## 玩法建议
    - **看最优策略绕开熔岩**：默认 VI,直接秒出最优策略,Mo 会漂亮地绕开所有禁区熔岩到达目标。
    - **看训练不足的代价**：选 Q-learning / Sarsa / MC ε-贪心,把 episode 数调很小（如 1000）或
      把 ε 调到 0,大概率看到 Mo **走进熔岩**或**在原地打转** —— 加 episode / 调大 ε 再看它学会。
    - **看奖励塑形**：把 r_forbidden 调到 −5,策略会更坚决地远离熔岩;把 r_step 设成略负（如 −0.1）,
      "尽快到达"成为唯一最优,路径最干净。
    - **墙 vs 熔岩**：墙(🧱/💧)**彻底挡路**,Mo 撞上去会弹开;熔岩**可踩但致命**。两者对比很直观:
      最优策略绕开墙、也躲开熔岩。**随机迷宫 🎲**:拖种子换布局(保证有解),用「迷宫熔岩数」加熔岩
      —— 熔岩只撒在最短路之外,所以 Mo 总有一条干净路线,你能看它同时躲墙又躲熔岩。想固定摆放就用"手动摆墙"。
    - **换个角色**：把角色切成 Rover(悬浮无人机),同一条策略换个形象走。
    - **对照传统表格**：这套世界、算法和隔壁 `custom_gridworld_lab.py` 完全一致 —— 那边是箭头表格,
      这边是 3D 行走,同一个策略两种看法。
    """, r"""
    ## Tips for playing
    - **Watch the optimal policy route around lava**: the default VI instantly yields the
      optimal policy, and Mo elegantly skirts every forbidden lava cell to the goal.
    - **See the cost of under-training**: pick Q-learning / Sarsa / MC ε-greedy, set the
      episode count very low (e.g. 1000) or ε to 0, and you'll likely watch Mo **walk into
      the lava** or **loop in place** — raise episodes / ε and watch it learn.
    - **See reward shaping**: set r_forbidden to −5 and the policy avoids lava more firmly;
      set r_step slightly negative (e.g. −0.1) so "reach fast" is the unique optimum and the
      path is cleanest.
    - **Swap the character**: switch to Rover (a hovering drone) to walk the same policy as a
      different figure.
    - **Compare with the flat table**: the worlds and algorithms are identical to the sibling
      `custom_gridworld_lab.py` — arrows-in-a-table there, a 3D walk here: one policy, two views.
    """))
    return


@app.cell(hide_code=True)
def _(ACTIONS, greedy_policy_from_q):
    # ---- bridge: GridWorld rollout -> Wanderland puzzle + egocentric commands ----
    # Wanderland heading ints: 0=N 1=E 2=S 3=W, matching gridworld's move deltas.
    # turn_right = clockwise (h -> h+1), turn_left = counter-clockwise (h -> h-1).
    _DELTA_TO_H = {(0, -1): 0, (1, 0): 1, (0, 1): 2, (-1, 0): 3}
    _H_NAME = {0: "N", 1: "E", 2: "S", 3: "W"}

    def rollout_greedy(_env, _Q, _max_steps):
        """Follow the greedy (argmax-Q) policy from start; return (states, actions).

        Stops early when a state is revisited (a looping policy) so the walk stays
        short and readable instead of wandering to the step cap."""
        _pol = greedy_policy_from_q(_Q).argmax(1)
        _env.reset()
        _s = _env.s_of(*_env.start)
        _tgt = _env.s_of(*_env.target)
        _states, _acts, _seen = [_s], [], {_s}
        if _s == _tgt:
            return _states, _acts
        for _ in range(_max_steps):
            _a = int(_pol[_s])
            _ns, _r, _done, _ = _env.step(_a)
            _acts.append(_a)
            _states.append(_ns)
            _s = _ns
            if _done or _s in _seen:
                break
            _seen.add(_s)
        return _states, _acts

    def traj_to_commands(_env, _acts):
        """Turn a list of absolute-direction action indices into egocentric verb
        names for World.act(), plus the initial heading name (so Mo starts facing
        the first move and wastes no spin). 'stay' (a=4) is dropped."""
        _init_h = 2  # default face South
        for _a in _acts:
            _d = _DELTA_TO_H.get(tuple(ACTIONS[_a]))
            if _d is not None:
                _init_h = _d
                break
        _h = _init_h
        _names = []
        for _a in _acts:
            _d = _DELTA_TO_H.get(tuple(ACTIONS[_a]))
            if _d is None:      # a == stay: no movement verb
                continue
            _diff = (_d - _h) % 4
            if _diff == 1:
                _names.append("turn_right")
            elif _diff == 2:
                _names += ["turn_right", "turn_right"]
            elif _diff == 3:
                _names.append("turn_left")
            _names.append("move_forward")
            _h = _d
        return _H_NAME[_init_h], _names

    def build_puzzle(_env, _Puzzle, _init_heading, _skin="walls"):
        """A Wanderland Puzzle mirroring the (Maze)GridWorld: target->goal,
        forbidden->lava, impassable walls-> wall blocks or a water moat (_skin)."""
        _obstacles = [tuple(_c) for _c in getattr(_env, "wall_set", set())]
        _kw = {"gaps": _obstacles} if _skin == "water" else {"walls": _obstacles}
        return _Puzzle(
            name="RL Grid",
            cols=_env.cols, rows=_env.rows,
            start=tuple(_env.start),
            heading=_init_heading,
            actions=("move_forward", "turn_left", "turn_right"),
            goal=tuple(_env.target),
            lava=[tuple(_c) for _c in _env.forbidden_set],
            **_kw,
        )

    return build_puzzle, rollout_greedy, traj_to_commands


@app.cell(hide_code=True)
def _(np):
    # ---- five methods (episodic: entering the target terminates) ----
    # Copied verbatim from custom_gridworld_lab.py so the two labs stay in lockstep.
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
        _nS, _nA = _env.n_states, _env.n_actions
        _tgt = _env.s_of(*_env.target)
        _pol = np.zeros(_nS, dtype=int)
        _Q = np.zeros((_nS, _nA))
        _n_iters = 10
        _per = max(1, _episodes // (_n_iters * _nS * _nA))
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
        _rng = np.random.default_rng(_seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _Q = np.zeros((_nS, _nA))
        _N = np.zeros((_nS, _nA))
        _pol = np.zeros(_nS, dtype=int)
        for _ep in range(_episodes):
            _cs = int(_rng.integers(_nS))
            _a = int(_rng.integers(_nA))
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
        _rng = np.random.default_rng(_seed)
        _nS, _nA = _env.n_states, _env.n_actions
        _Q = np.zeros((_nS, _nA))
        _N = np.zeros((_nS, _nA))
        for _ep in range(_episodes):
            _cs = _env.reset()
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
    from matplotlib import patches
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False

    def _ensure_cjk_font():
        """Make the Chinese plot labels render. A desktop (Windows/macOS) already has a
        CJK font; a bare Linux kernel (molab) has none and would draw tofu boxes, so
        fetch one once and register it. Failure is swallowed — worst case: boxes."""
        from matplotlib import font_manager

        _installed = {f.name for f in font_manager.fontManager.ttflist}
        if _installed & {"Microsoft YaHei", "SimHei", "PingFang SC",
                         "Noto Sans CJK SC", "WenQuanYi Micro Hei"}:
            return
        try:
            import os
            import tempfile
            import urllib.request

            _p = os.path.join(tempfile.gettempdir(), "wqy-microhei.ttc")
            if not os.path.exists(_p):
                urllib.request.urlretrieve(
                    "https://cdn.jsdelivr.net/gh/anthonyfok/fonts-wqy-microhei/"
                    "wqy-microhei.ttc", _p)
            font_manager.fontManager.addfont(_p)
            matplotlib.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
        except Exception:
            pass

    _ensure_cjk_font()

    # Wanderland is a published PyPI package, declared in the PEP 723 header above so
    # marimo.app / molab install it automatically. Locally: `pip install wanderland`.
    try:
        from wanderland import World, Puzzle
    except ModuleNotFoundError as _e:
        raise ModuleNotFoundError(
            "This lab needs the 'wanderland' package — install it with: "
            "pip install wanderland"
        ) from _e

    return Puzzle, World, np, patches, plt


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(np, patches, plt):
    # ---- self-contained grid world + maze extension (inlined from
    #      foundations/gridworld.py so this lab runs standalone in molab / WASM,
    #      with no sibling-module dependency and no name clash with the PyPI
    #      'gridworld' package) ----
    from dataclasses import dataclass, field
    from collections import deque

    ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]  # up, right, down, left, stay

    @dataclass
    class GridWorld:
        cols: int = 5
        rows: int = 5
        start: tuple = (0, 0)
        target: tuple = (2, 3)
        forbidden: tuple = ()
        r_target: float = 1.0
        r_forbidden: float = -1.0
        r_step: float = 0.0
        r_boundary: float = -1.0
        gamma: float = 0.9
        forbidden_set: set = field(init=False)

        def __post_init__(self):
            self.forbidden_set = set(map(tuple, self.forbidden))
            self.n_states = self.rows * self.cols
            self.n_actions = len(ACTIONS)
            self.agent = self.start

        def s_of(self, x, y):
            return y * self.cols + x

        def xy_of(self, s):
            return (s % self.cols, s // self.cols)

        def in_bounds(self, x, y):
            return 0 <= x < self.cols and 0 <= y < self.rows

        def _cell_reward(self, x, y):
            if (x, y) == tuple(self.target):
                return self.r_target
            if (x, y) in self.forbidden_set:
                return self.r_forbidden
            return self.r_step

        def transition(self, s, a):
            x, y = self.xy_of(s)
            dx, dy = ACTIONS[a]
            nx, ny = x + dx, y + dy
            if not self.in_bounds(nx, ny):
                return s, self.r_boundary
            return self.s_of(nx, ny), self._cell_reward(nx, ny)

        def build_model(self):
            P = np.zeros((self.n_states, self.n_actions, self.n_states))
            R = np.zeros((self.n_states, self.n_actions))
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    ns, r = self.transition(s, a)
                    P[s, a, ns] = 1.0
                    R[s, a] = r
            return P, R

        def reset(self, state=None):
            if state is None:
                self.agent = self.start
            elif isinstance(state, (int, np.integer)):
                self.agent = self.xy_of(int(state))
            else:
                self.agent = tuple(state)
            return self.s_of(*self.agent)

        def step(self, a):
            s = self.s_of(*self.agent)
            ns, r = self.transition(s, a)
            self.agent = self.xy_of(ns)
            done = self.xy_of(ns) == tuple(self.target)
            return ns, r, done, {}

        def _base_ax(self, ax, title=None):
            ax.set_xlim(-0.5, self.cols - 0.5)
            ax.set_ylim(-0.5, self.rows - 0.5)
            ax.set_xticks(np.arange(-0.5, self.cols, 1))
            ax.set_yticks(np.arange(-0.5, self.rows, 1))
            ax.grid(True, color="gray", linewidth=1)
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.xaxis.set_ticks_position("top")
            ax.tick_params(labelbottom=False, labelleft=False, labeltop=False,
                           bottom=False, left=False, right=False, top=False)
            for (fx, fy) in self.forbidden_set:
                ax.add_patch(patches.Rectangle((fx - 0.5, fy - 0.5), 1, 1,
                             facecolor=(0.929, 0.694, 0.125)))
            tx, ty = self.target
            ax.add_patch(patches.Rectangle((tx - 0.5, ty - 0.5), 1, 1,
                         facecolor=(0.301, 0.745, 0.933)))
            if title:
                ax.set_title(title, pad=22)

        def plot(self, title=None, figsize=(4.4, 4.4)):
            fig, ax = plt.subplots(figsize=figsize)
            self._base_ax(ax, title)
            return fig, ax

    def greedy_policy_from_q(_Q):
        """Deterministic (one-hot) policy greedy w.r.t. action values Q."""
        _pi = np.zeros_like(_Q)
        _pi[np.arange(_Q.shape[0]), _Q.argmax(axis=1)] = 1.0
        return _pi

    # MazeGridWorld: walls as truly impassable cells (distinct from lava). Only
    # in_bounds() is overridden (entering a wall == the grid edge: stay + r_boundary);
    # transition()/build_model()/step() all route through it, so every algorithm
    # respects walls automatically.
    @dataclass
    class MazeGridWorld(GridWorld):
        walls: tuple = ()
        wall_set: set = field(init=False, default_factory=set)

        def __post_init__(self):
            super().__post_init__()
            self.wall_set = set(map(tuple, self.walls))

        def in_bounds(self, x, y):
            return super().in_bounds(x, y) and (x, y) not in self.wall_set

    def generate_maze(_cols, _rows, _start, _target, _seed, _density=0.28):
        """A random *solvable* maze: add walls one at a time, keeping each only while a
        start->target path still exists (BFS). Deterministic in _seed."""
        _rng = np.random.default_rng(_seed)
        _start, _target = tuple(_start), tuple(_target)
        _cells = [(x, y) for y in range(_rows) for x in range(_cols)
                  if (x, y) not in (_start, _target)]
        _order = _rng.permutation(len(_cells))
        _walls, _budget = set(), int(_density * _cols * _rows)

        def _connected():
            _seen = {_start}
            _dq = deque([_start])
            while _dq:
                x, y = _dq.popleft()
                if (x, y) == _target:
                    return True
                for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < _cols and 0 <= ny < _rows
                            and (nx, ny) not in _walls and (nx, ny) not in _seen):
                        _seen.add((nx, ny))
                        _dq.append((nx, ny))
            return False

        for _i in _order:
            if len(_walls) >= _budget:
                break
            _c = _cells[_i]
            _walls.add(_c)
            if not _connected():
                _walls.discard(_c)
        return tuple(sorted(_walls))

    def random_lava(_cols, _rows, _start, _target, _walls, _n, _seed):
        """Scatter _n lava cells that are OFF a shortest start->target path, so the
        optimal policy always has a clean lava-free route (Mo never has to die) while
        still weaving between hazards. Deterministic in _seed."""
        if _n <= 0:
            return ()
        _start, _target = tuple(_start), tuple(_target)
        _wset = set(_walls)
        # BFS shortest path (with parent pointers) to protect its cells from lava
        _prev = {_start: None}
        _dq = deque([_start])
        while _dq:
            _cur = _dq.popleft()
            if _cur == _target:
                break
            _x, _y = _cur
            for _dx, _dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                _nc = (_x + _dx, _y + _dy)
                if (0 <= _nc[0] < _cols and 0 <= _nc[1] < _rows
                        and _nc not in _wset and _nc not in _prev):
                    _prev[_nc] = _cur
                    _dq.append(_nc)
        _path, _c = set(), _target
        while _c is not None:
            _path.add(_c)
            _c = _prev.get(_c)
        _cand = [(x, y) for y in range(_rows) for x in range(_cols)
                 if (x, y) not in _wset and (x, y) not in _path]
        if not _cand:
            return ()
        _rng = np.random.default_rng(_seed + 10007)
        _idx = _rng.permutation(len(_cand))[:min(_n, len(_cand))]
        return tuple(sorted(_cand[_i] for _i in _idx))

    return ACTIONS, MazeGridWorld, generate_maze, greedy_policy_from_q, random_lava


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
