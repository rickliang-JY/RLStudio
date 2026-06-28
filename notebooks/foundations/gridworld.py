"""Grid World environment for "Mathematical Foundation of Reinforcement Learning"
(Shiyu Zhao).

This is a self-contained, dependency-light re-implementation of the book's grid
world, designed to be shared by every chapter notebook. It exposes BOTH:

  * a *model* (transition probabilities P and expected rewards R) for the
    model-based methods of chapters 2-4, and
  * a *sampling* interface (`step`) for the model-free methods of chapters 5-10.

Conventions (identical to the book's figures)
---------------------------------------------
* The grid has `rows` x `cols` cells. A cell is addressed by (x, y) where x is the
  column (0..cols-1, left->right) and y is the row (0..rows-1, top->bottom).
* A state index is  s = y * cols + x.
* Action order:  0=up, 1=right, 2=down, 3=left, 4=stay.
* Dynamics are deterministic. Moving off the grid keeps the agent in place and
  yields `r_boundary`. Otherwise the agent moves to the neighbouring cell and the
  reward depends only on the cell it lands in: target -> `r_target`,
  forbidden -> `r_forbidden`, otherwise `r_step`. Forbidden cells ARE enterable
  (the book penalises them with a reward, it does not wall them off).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Render Chinese plot titles/labels correctly on Windows.
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# action index -> (dx, dy).  y grows downward (row index), so dy=+1 is "down".
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
ACTION_NAMES = ["up", "right", "down", "left", "stay"]


@dataclass
class GridWorld:
    cols: int = 5
    rows: int = 5
    start: tuple = (0, 0)
    target: tuple = (2, 3)
    forbidden: tuple = ((1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4))
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
        self.agent = self.start  # mutable cursor used by reset/step

    # ---- index helpers ---------------------------------------------------
    def s_of(self, x, y) -> int:
        return y * self.cols + x

    def xy_of(self, s) -> tuple:
        return (s % self.cols, s // self.cols)

    def in_bounds(self, x, y) -> bool:
        return 0 <= x < self.cols and 0 <= y < self.rows

    def _cell_reward(self, x, y) -> float:
        if (x, y) == tuple(self.target):
            return self.r_target
        if (x, y) in self.forbidden_set:
            return self.r_forbidden
        return self.r_step

    # ---- one-step deterministic dynamics ---------------------------------
    def transition(self, s, a):
        """Return (next_state_index, reward) for taking action `a` in state `s`."""
        x, y = self.xy_of(s)
        dx, dy = ACTIONS[a]
        nx, ny = x + dx, y + dy
        if not self.in_bounds(nx, ny):
            return s, self.r_boundary           # bounce off the wall, stay put
        return self.s_of(nx, ny), self._cell_reward(nx, ny)

    # ---- full model (used by model-based chapters 2-4) -------------------
    def build_model(self):
        """Return (P, R): P[s,a,s'] in {0,1}, R[s,a] expected immediate reward."""
        P = np.zeros((self.n_states, self.n_actions, self.n_states))
        R = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ns, r = self.transition(s, a)
                P[s, a, ns] = 1.0
                R[s, a] = r
        return P, R

    # ---- sampling interface (used by model-free chapters 5-10) ----------
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

    # =====================================================================
    #  Visualisation helpers — every one returns a matplotlib Figure so it
    #  renders inline in marimo.
    # =====================================================================
    def _base_ax(self, ax, title=None):
        ax.set_xlim(-0.5, self.cols - 0.5)
        ax.set_ylim(-0.5, self.rows - 0.5)
        ax.set_xticks(np.arange(-0.5, self.cols, 1))
        ax.set_yticks(np.arange(-0.5, self.rows, 1))
        ax.grid(True, color="gray", linewidth=1)
        ax.set_aspect("equal")
        ax.invert_yaxis()                 # row 0 at the top
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(labelbottom=False, labelleft=False, labeltop=False,
                       bottom=False, left=False, right=False, top=False)
        for x in range(self.cols):
            ax.text(x, -0.72, str(x + 1), ha="center", va="center", fontsize=9)
        for y in range(self.rows):
            ax.text(-0.72, y, str(y + 1), ha="center", va="center", fontsize=9)
        # colour the special cells
        for (fx, fy) in self.forbidden_set:
            ax.add_patch(patches.Rectangle((fx - 0.5, fy - 0.5), 1, 1,
                         facecolor=(0.929, 0.694, 0.125)))
        tx, ty = self.target
        ax.add_patch(patches.Rectangle((tx - 0.5, ty - 0.5), 1, 1,
                     facecolor=(0.301, 0.745, 0.933)))
        if title:
            ax.set_title(title, pad=22)

    def plot(self, title=None, figsize=(4.2, 4.2)):
        fig, ax = plt.subplots(figsize=figsize)
        self._base_ax(ax, title)
        return fig, ax

    def _draw_policy(self, ax, policy):
        """policy: array (n_states, n_actions) of action probabilities."""
        col = (0.466, 0.674, 0.188)
        for s in range(self.n_states):
            x, y = self.xy_of(s)
            for a, p in enumerate(policy[s]):
                if p <= 1e-6:
                    continue
                dx, dy = ACTIONS[a]
                if (dx, dy) == (0, 0):
                    ax.add_patch(patches.Circle((x, y), 0.08, fill=False,
                                 edgecolor=col, linewidth=1.5))
                else:
                    L = 0.1 + 0.3 * p
                    ax.add_patch(patches.FancyArrow(x, y, L * dx, L * dy,
                                 color=col, width=0.02, head_width=0.12,
                                 length_includes_head=True))

    def plot_policy(self, policy, values=None, title=None, precision=1,
                    figsize=(4.6, 4.6)):
        fig, ax = self.plot(title=title, figsize=figsize)
        self._draw_policy(ax, np.asarray(policy))
        if values is not None:
            v = np.round(np.asarray(values), precision)
            for s in range(self.n_states):
                x, y = self.xy_of(s)
                ax.text(x, y + 0.32, f"{v[s]:.{precision}f}", ha="center",
                        va="center", fontsize=7, color="black")
        return fig

    def plot_values(self, values, title=None, precision=2, figsize=(4.6, 4.6),
                    cmap="viridis"):
        fig, ax = self.plot(title=title, figsize=figsize)
        v = np.asarray(values, dtype=float).reshape(self.rows, self.cols)
        ax.imshow(v, cmap=cmap, alpha=0.55, origin="upper",
                  extent=(-0.5, self.cols - 0.5, self.rows - 0.5, -0.5))
        for s in range(self.n_states):
            x, y = self.xy_of(s)
            ax.text(x, y, f"{values[s]:.{precision}f}", ha="center",
                    va="center", fontsize=8, color="black")
        return fig

    def plot_trajectory(self, traj, title=None, figsize=(4.6, 4.6),
                        mark_current=False):
        """traj: list of state indices visited.

        Jitter is deterministic (seeded by the state index) so that scrubbing a
        step slider does not make the path wobble between renders. When
        `mark_current` is set, the last visited cell is highlighted in red so you
        can see where the agent is *right now*.
        """
        fig, ax = self.plot(title=title, figsize=figsize)
        xs, ys = zip(*[self.xy_of(s) for s in traj])
        rng = np.random.default_rng(0)
        jx = 0.06 * rng.standard_normal(self.n_states)
        jy = 0.06 * rng.standard_normal(self.n_states)
        xs = np.array(xs) + jx[list(traj)]
        ys = np.array(ys) + jy[list(traj)]
        ax.plot(xs, ys, "-o", color=(0, 0.4, 0), markersize=4, linewidth=1)
        ax.plot(xs[0], ys[0], "*", color="blue", markersize=18)
        if mark_current and len(xs) > 0:
            ax.plot(xs[-1], ys[-1], "o", color="red", markersize=12,
                    markerfacecolor="none", markeredgewidth=2.5)
        return fig


def greedy_policy_from_q(Q):
    """Deterministic (one-hot) policy that is greedy w.r.t. action values Q."""
    n_states, n_actions = Q.shape
    pi = np.zeros_like(Q)
    pi[np.arange(n_states), Q.argmax(axis=1)] = 1.0
    return pi


def q_from_v(v, P, R, gamma):
    """Action values q(s,a) given state values v and the model."""
    return R + gamma * P.dot(v)


# The canonical 5x5 grid used throughout Zhao's book (e.g. Figures 5.6-5.8):
# target at column 3 / row 4, with the L-shaped wall of forbidden cells around it.
# Rewards follow the book's standard setting: r_boundary = r_forbidden = -1,
# r_target = 1, r_other = 0 (no per-step cost), gamma = 0.9.
def classic_example() -> GridWorld:
    return GridWorld(
        cols=5, rows=5, start=(0, 0), target=(2, 3),
        forbidden=((1, 1), (2, 1), (2, 2), (1, 3), (3, 3), (1, 4)),
        r_target=1.0, r_forbidden=-1.0, r_step=0.0, r_boundary=-1.0, gamma=0.9,
    )
