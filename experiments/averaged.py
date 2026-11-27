"""
averaged.py - every figure of the paper again, but AVERAGED OVER SEVERAL
RANDOM SEEDS, drawn as clean mean curves with a light band.

Why: a single run (seed 0, the rest of this replication) gives noisy curves
that depend on the network's random starting weights and on which samples
the replay memory happens to draw. Here each setting is run with seeds
0, 1, ..., S-1 (default S = 10). For every seed we take the usual moving
average, then plot

    line = mean over the seeds,
    band = +/- 1 standard error of that mean (std over seeds / sqrt(seeds))

so the line shows the typical behaviour and the band how precisely the
average is known. The run-to-run spread itself (std, worst and best seed)
is listed per setting in results/averaged/summary.md. The parameter sweeps
(Figs. 9-12) use a 500-frame moving average instead of the paper's 200, so
the lines are smooth enough to compare at a glance. The channel data is the authors' (the same for every
seed); only the DROO randomness changes.

What is the same as the single-seed scripts: the settings of every figure
(N, K, Delta, memory, batch, interval, learning rate, weight/WD schedules),
the authors' MemoryDNN and decoder, and the data. What is new:
  - solver='vectorized' (baselines/p2_vectorized.py): the authors' (P2)
    equations solved for all K candidates at once. It picks exactly the same
    actions as bisection() (checked: same decisions, rates equal to ~1e-9),
    and makes a run ~8x faster, which is what makes 10 seeds affordable.
    Exception: the Fig. 11-13 runs keep the authors' bisection(), because
    Fig. 13's x-axis is the CPU time, which must be the original algorithm's.
  - Fig. 9's four panels are drawn as one 2x2 figure.
  - Table 2 timings are not repeated (timing does not depend on the seed);
    Fig. 14/15 use the DROO rates of all seeds.

Runs that share a setting are run once: the paper's default setting
(N=10, fixed K=10, memory 1024, batch 128, interval 10, lr 0.01) gives
Fig. 4, Fig. 5, the default line of Fig. 9, "OP, K=10" in Fig. 10 and
"Delta = infinity" in Fig. 11/12.

How to run (from the project folder):
    python experiments/averaged.py                  # all runs, then all figures
    python experiments/averaged.py --only fig09     # only the runs of one figure
    python experiments/averaged.py --seeds 5        # fewer seeds
    python experiments/averaged.py --plot-only      # redraw from saved runs
    python experiments/averaged.py --dry-run        # list the jobs

Output: results/averaged/<setting>/seed<k>.npz (+ _summary.json),
        figures/averaged/*.png, results/averaged/summary.md
Safe to stop and restart: finished runs are skipped.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
from scipy.io import loadmat      # noqa: E402

from common import (FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR, frames,   # noqa: E402
                    load_data, run_droo)
from p2_vectorized import solve_p2                                    # noqa: E402

OUT = RESULTS_DIR / "averaged"
FIG = FIGURES_DIR / "averaged"
N = 10
FRAMES = frames(10000)
TABLE2_FRAMES = frames(30000, 500)

# ---------------------------------------------------------------------------
# Hooks (same schedules as fig06.py, fig07.py, fig08.py)
# ---------------------------------------------------------------------------
SWAP_FRAMES = [int(0.6 * FRAMES), int(0.8 * FRAMES)]        # Fig. 6
T_WD2, T_WD1, T_RESET = int(.4 * FRAMES), int(.6 * FRAMES), int(.8 * FRAMES)  # Fig. 7
W_DEFAULT = np.array([1, 1.5] * (N // 2), dtype=float)
W_A = W_DEFAULT.copy()
W_A[1] = 3.0
W_B = W_A.copy()
W_B[0] = 3.0
ONOFF = [(0.60, "off", 1), (0.65, "off", 1), (0.70, "off", 1), (0.75, "off", 1),
         (0.80, "on", 1), (0.85, "on", 1), (0.90, "on", 2), (0.95, "off", 2)]
ONOFF_EVENTS = {int(f * FRAMES): (a, c) for f, a, c in ONOFF}   # Fig. 8


def fig07_weights_at(i):
    if T_WD2 <= i < T_WD1:
        return W_A
    if T_WD1 <= i < T_RESET:
        return W_B
    return W_DEFAULT


def make_hook(kind):
    if kind == "swap":                                         # Fig. 6
        _, rate_def = load_data("data_10")
        _, rate_swp = load_data("data_10_WeightsAlternated")
        dw, sw = [1, 1.5] * (N // 2), [1.5, 1] * (N // 2)

        def hook(i, state):
            if i == 0:
                state["weights"], state["rate"] = dw, rate_def
            elif i == SWAP_FRAMES[0]:
                state["weights"], state["rate"] = sw, rate_swp
            elif i == SWAP_FRAMES[1]:
                state["weights"], state["rate"] = dw, rate_def
        return hook

    if kind == "new_weights":                                  # Fig. 7
        ref = np.load(PROJECT_ROOT / "results" / "fig07" / "reference.npz")["rate"][:, None]

        def hook(i, state):
            if i == 0:
                state["weights"], state["rate"] = list(W_DEFAULT), ref
            if i in (T_WD2, T_WD1, T_RESET):
                state["weights"] = list(fig07_weights_at(i))
        return hook

    if kind == "onoff":                                        # Fig. 8
        def hook(i, state):
            if i not in ONOFF_EVENTS:
                return
            action, count = ONOFF_EVENTS[i]
            for _ in range(count):
                if action == "off" and state["N_active"] > 5:
                    state["N_active"] -= 1
                    k = state["N_active"]
                    state["channel"][:, k] = state["channel"][:, k] / 1000000
                elif action == "on" and state["N_active"] < N:
                    state["N_active"] += 1
                    k = state["N_active"] - 1
                    state["channel"][:, k] = state["channel"][:, k] * 1000000
            _, state["rate"] = load_data("data_%d" % state["N_active"])
        return hook
    return None


# ---------------------------------------------------------------------------
# Every distinct setting: name -> run_droo arguments
# ---------------------------------------------------------------------------
BASE = dict(N=N, n=FRAMES, K=N, delta=None, memory=1024, batch_size=128,
            train_interval=10, lr=0.01, decoder="OP")
SETTINGS = {"default": dict(BASE)}
SETTINGS["fig06_swap"] = dict(BASE, split=False, hook="swap")
SETTINGS["fig07_new_weights"] = dict(BASE, split=False, hook="new_weights")
SETTINGS["fig08_onoff"] = dict(BASE, split=False, hook="onoff")

FIG09 = {"memory": ("memory", [128, 256, 512, 1024, 2048], "Memory size"),
         "batch": ("batch_size", [32, 64, 128, 512, 1024], "Batch size"),
         "interval": ("train_interval", [5, 10, 20, 50, 100], "Training interval"),
         "lr": ("lr", [0.1, 0.01, 0.001, 0.0001, 0.00001], "Learning rate")}


def fig09_name(panel, value):
    arg = FIG09[panel][0]
    return "default" if value == BASE[arg] else "fig09_%s_%s" % (panel, value)


for panel, (arg, values, _) in FIG09.items():
    for v in values:
        if fig09_name(panel, v) != "default":
            SETTINGS[fig09_name(panel, v)] = dict(BASE, **{arg: v})

FIG10 = [("OP", 10), ("KNN", 10), ("OP", 2), ("KNN", 2), ("OP", 1), ("KNN", 1)]


def fig10_name(dec, k):
    return "default" if (dec, k) == ("OP", 10) else "fig10_%s_K%d" % (dec, k)


for dec, k in FIG10:
    if fig10_name(dec, k) != "default":
        SETTINGS[fig10_name(dec, k)] = dict(BASE, decoder=dec, K=k)

DELTAS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
FIG11_12 = [1, 2, 4, 16, 128, 1024, None]        # None = infinity (fixed K)
for d in DELTAS:
    # bisection: Fig. 13 plots CPU time, which must be the authors' solver's
    SETTINGS["fig11_delta_%d" % d] = dict(BASE, delta=d, solver="bisection")

NS = [10, 20, 30]
for n_wd in NS:
    SETTINGS["table2_droo_N%d" % n_wd] = dict(BASE, N=n_wd, K=n_wd, delta=32,
                                             n=TABLE2_FRAMES)

GROUPS = {   # --only <group>
    "fig04": ["default"], "fig05": ["default"], "fig06": ["fig06_swap"],
    "fig07": ["fig07_new_weights"], "fig08": ["fig08_onoff"],
    "fig09": sorted({fig09_name(p, v) for p in FIG09 for v in FIG09[p][1]}),
    "fig10": sorted({fig10_name(d, k) for d, k in FIG10}),
    "fig11_13": ["fig11_delta_%d" % d for d in DELTAS] + ["default"],
    "table2_fig14_15": ["table2_droo_N%d" % n for n in NS],
}


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------
def run_path(name, seed):
    return OUT / name / ("seed%d.npz" % seed)


def run_job(name, seed):
    import torch
    torch.set_num_threads(1)
    cfg = dict(SETTINGS[name])
    hook = make_hook(cfg.pop("hook", None))
    cfg.setdefault("solver", "vectorized")
    r = run_droo(seed=seed, hook=hook, verbose=False, **cfg)
    keep = {"rate_ratio": r["rate_ratio"].astype(np.float32),
            "K": r["K"].astype(np.int8),
            "loss": np.asarray(r["loss"], dtype=np.float32)}
    if name == "default":
        keep["k_idx"] = r["k_idx"].astype(np.int8)                 # Fig. 4
    if name.startswith("table2"):
        nt = r["config"]["num_test"]
        keep["rate_test"] = r["rate"][-nt:].astype(np.float64)      # Fig. 14
    if name == "fig07_new_weights":
        keep["relative_wd12"] = fig07_relative(r["mode"]).astype(np.float32)
    path = run_path(name, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **keep)
    with open(path.with_name("seed%d_summary.json" % seed), "w") as f:
        json.dump({"config": r["config"], "summary": r["summary"]}, f, indent=2)
    return r["summary"]["total_time_s"]


def fig07_relative(modes):
    """Each of WD1/WD2's rate under DROO's action, divided by its rate under
    the optimal action for the original weights (as in fig07.py)."""
    channel, _ = load_data("data_10")
    h = channel / 1000000
    best = loadmat(str(PROJECT_ROOT / "DROO" / "data" / "data_10"))["output_mode"]
    rel = np.zeros((len(modes), 2))
    for i in range(len(modes)):
        droo = solve_p2(h[i], modes[i], fig07_weights_at(i))["r_wd"][0]
        base = solve_p2(h[i], best[i], W_DEFAULT)["r_wd"][0]
        rel[i] = droo[:2] / base[:2]
    return rel


# ---------------------------------------------------------------------------
# Loading and averaging
# ---------------------------------------------------------------------------
def load_runs(name, seeds):
    runs = []
    for s in seeds:
        p = run_path(name, s)
        if p.exists():
            d = dict(np.load(p))
            with open(p.with_name("seed%d_summary.json" % s)) as f:
                d.update(json.load(f))
            runs.append(d)
    if not runs:
        raise FileNotFoundError("no saved runs for %s" % name)
    return runs


def movavg(x, window):
    """Trailing moving average (same as common.rolling's mean, but fast)."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(np.insert(x, 0, 0.0))
    t = np.arange(1, len(x) + 1)
    lo = np.maximum(0, t - window)
    return (c[t] - c[lo]) / (t - lo)


def seed_band(runs, key, window, col=None):
    """Moving average of `key` for every seed -> (mean, standard error) over seeds."""
    curves = []
    for r in runs:
        x = r[key] if col is None else r[key][:, col]
        curves.append(movavg(x, window))
    curves = np.vstack(curves)
    n = len(curves)
    return curves.mean(axis=0), curves.std(axis=0) / np.sqrt(n), n


# ---------------------------------------------------------------------------
# Plot style (one look for every figure)
# ---------------------------------------------------------------------------
COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17",
          "#6a4ce2", "#ec4647"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 10,
    "axes.edgecolor": "#b3b3b3", "axes.labelcolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "legend.fontsize": 8.5, "lines.linewidth": 1.6,
})


def band_line(ax, mean, std, color, label, style="-", alpha=0.2):
    t = np.arange(1, len(mean) + 1)
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=alpha, linewidth=0)
    ax.plot(t, mean, style, color=color, label=label)


def note(ax, n_seeds, extra=""):
    ax.text(0.99, 0.02, "mean of %d seeds, band = ±1 standard error%s" % (n_seeds, extra),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=INK2)


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved", path)


def mark_events(axes, events, top_ax):
    for frame, label in events:
        for ax in axes:
            ax.axvline(frame, color="#999999", linestyle="--", linewidth=0.8)
        top_ax.annotate(label, (frame, 1.0), xycoords=("data", "axes fraction"),
                        fontsize=8, ha="center", va="bottom", color=INK2,
                        annotation_clip=False)


def loss_rate_figure(name, title, seeds, events=(), fname=None, rate_ylim=(0.9, 1.0)):
    """Fig. 5/6/8 style: training loss on top, normalized rate below."""
    runs = load_runs(name, seeds)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    m, s, n = seed_band(runs, "loss", 20)
    interval = runs[0]["config"]["train_interval"]
    t = np.arange(1, len(m) + 1) * interval
    ax1.fill_between(t, m - s, m + s, color=COLORS[7], alpha=0.15, linewidth=0)
    ax1.plot(t, m, color=COLORS[7])
    ax1.set_ylabel("Training loss")
    m, s, n = seed_band(runs, "rate_ratio", 50)
    band_line(ax2, m, s, COLORS[0], None)
    ax2.set_ylim(*rate_ylim)
    ax2.set_ylabel("Normalized computation rate\n(50-frame moving average)")
    ax2.set_xlabel("Time frame t")
    note(ax2, n)
    ax2.texts[-1].set_position((0.12, 0.03))
    ax2.texts[-1].set_ha("left")
    mark_events((ax1, ax2), events, ax1)
    fig.suptitle(title, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, fname)


SWEEP_WINDOW = 500


def sweep_axis(ax, lines, seeds, window=SWEEP_WINDOW, ylim=None, ylabel=None, key="rate_ratio"):
    """Several settings on one axis: mean line + band for each."""
    n = 0
    for k, (label, name, style) in enumerate(lines):
        m, s, n = seed_band(load_runs(name, seeds), key, window)
        band_line(ax, m, s, COLORS[k], label, style)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel or "Normalized rate (%d-frame moving avg.)" % window)
    return n


# ---------------------------------------------------------------------------
# The figures
# ---------------------------------------------------------------------------
def plot_all(seeds):
    S = list(seeds)
    tried = []

    def attempt(fn):
        try:
            fn()
        except FileNotFoundError as e:
            print("skipped %s: %s" % (fn.__name__, e))
        tried.append(fn.__name__)

    def fig04():
        runs = load_runs("default", S)
        m, s, n = seed_band([dict(r, kk=r["k_idx"] + 1.0) for r in runs], "kk", 50)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        band_line(ax, m, s, COLORS[6], None, alpha=0.2)
        ax.set_ylim(0.5, 10.5)
        ax.set_xlabel("Time frame t")
        ax.set_ylabel("Selected index k*_t (50-frame moving avg.)")
        ax.set_title("Fig. 4: rank of the best action among K = 10 candidates (N = 10)")
        note(ax, n)
        save(fig, "fig04_kstar_avg.png")

    def fig05():
        loss_rate_figure("default", "Fig. 5: convergence of DROO (N = 10, K = 10)", S,
                         fname="fig05_convergence_avg.png")

    def fig06():
        loss_rate_figure("fig06_swap", "Fig. 6: alternating WD weights (N = 10, K = 10)",
                         S, events=[(SWAP_FRAMES[0], "swap"), (SWAP_FRAMES[1], "reset")],
                         fname="fig06_alternate_weights_avg.png", rate_ylim=(0.85, 1.0))

    def fig08():
        ev = [(f, "%s x%d" % (a, c) if c > 1 else a) for f, (a, c) in sorted(ONOFF_EVENTS.items())]
        loss_rate_figure("fig08_onoff", "Fig. 8: WDs switched off and on (N = 10, K = 10)",
                         S, events=ev, fname="fig08_onoff_avg.png", rate_ylim=(0.85, 1.0))

    def fig07():
        runs = load_runs("fig07_new_weights", S)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.2), sharex=True)
        for k, (wd, c) in enumerate([("WD1", COLORS[0]), ("WD2", COLORS[1])]):
            m, s, n = seed_band(runs, "relative_wd12", 200, col=k)
            band_line(ax1, m, s, c, wd)
        ax1.axhline(1, color="#999999", linewidth=0.8)
        ax1.set_ylabel("Relative computation rate\n(200-frame moving avg.)")
        ax1.legend(loc="upper left")
        m, s, n = seed_band(runs, "rate_ratio", 50)
        band_line(ax2, m, s, COLORS[0], None)
        ax2.set_ylim(0.85, 1.0)
        ax2.set_ylabel("Normalized computation rate\n(50-frame moving avg.)")
        ax2.set_xlabel("Time frame t")
        note(ax2, n)
        ax2.texts[-1].set_position((0.12, 0.03))
        ax2.texts[-1].set_ha("left")
        mark_events((ax1, ax2), [(T_WD2, "WD2 x2"), (T_WD1, "WD1 x3"), (T_RESET, "reset")], ax1)
        fig.suptitle("Fig. 7: temporarily higher weights for WD1 and WD2 (N = 10, K = 10)",
                     fontweight="bold", y=1.0)
        fig.tight_layout()
        save(fig, "fig07_new_weights_avg.png")

    def fig09():
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        n = 0
        for ax, (panel, (arg, values, label)), letter in zip(axes.flat, FIG09.items(), "abcd"):
            lines = [("%s%s" % (v, " (default)" if v == BASE[arg] else ""),
                      fig09_name(panel, v), "-") for v in values]
            n = sweep_axis(ax, lines, S, ylim=(0.92, 1.0),
                           ylabel="Normalized rate (%d-frame moving avg.)" % SWEEP_WINDOW
                           if letter in "ac" else "")
            ax.set_title("(%s) %s" % (letter, label))
            ax.legend(title=label, title_fontsize=8.5, loc="center left",
                      bbox_to_anchor=(1.0, 0.5))
            if letter in "cd":
                ax.set_xlabel("Time frame t")
        fig.suptitle("Fig. 9: effect of the training parameters (N = 10, K = 10); "
                     "mean of %d seeds, band = ±1 standard error" % n, fontweight="bold")
        fig.tight_layout()
        save(fig, "fig09_parameters_avg.png")

    def fig10():
        names = {"OP": "Order-preserving", "KNN": "KNN"}
        fig, ax = plt.subplots(figsize=(8.5, 5))
        n = 0
        for k, kval in enumerate([10, 2, 1]):
            for dec, style in (("OP", "-"), ("KNN", "--")):
                runs = load_runs(fig10_name(dec, kval), S)
                m, s, n = seed_band(runs, "rate_ratio", SWEEP_WINDOW)
                band_line(ax, m, s, COLORS[k if dec == "OP" else k + 3],
                          "%s, K = %d" % (names[dec], kval), style)
        ax.set_ylim(0.65, 1.0)
        ax.set_xlabel("Time frame t")
        ax.set_ylabel("Normalized rate (%d-frame moving avg.)" % SWEEP_WINDOW)
        ax.set_title("Fig. 10: order-preserving vs KNN quantization (N = 10)")
        ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
        note(ax, n)
        ax.texts[-1].set_position((0.99, 0.80))
        ax.text(0.99, 0.75, "K = 1: both methods give the same candidate, so the lines coincide",
                transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
        save(fig, "fig10_quantization_avg.png")

    def d_name(d):
        return "default" if d is None else "fig11_delta_%d" % d

    def d_label(d):
        return "Δ = ∞ (fixed K)" if d is None else "Δ = %d" % d

    def fig11():
        fig, ax = plt.subplots(figsize=(8.5, 5))
        n = sweep_axis(ax, [(d_label(d), d_name(d), "--" if d == 1024 else "-")
                            for d in FIG11_12], S, ylim=(0.90, 1.0))
        ax.set_xlabel("Time frame t")
        ax.set_title("Fig. 11: adaptive K with different update intervals Δ (N = 10)")
        ax.legend(loc="lower right", ncol=2)
        note(ax, n)
        ax.texts[-1].set_position((0.99, 0.30))
        save(fig, "fig11_delta_rate_avg.png")

    def fig12():
        fig, ax = plt.subplots(figsize=(8.5, 5))
        n = sweep_axis(ax, [(d_label(d), d_name(d), "-") for d in FIG11_12], S,
                       ylim=(1, 10.5), ylabel="K_t (%d-frame moving avg.)" % SWEEP_WINDOW, key="K")
        ax.set_xlabel("Time frame t")
        ax.set_title("Fig. 12: number of candidates K_t under different Δ (N = 10)")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        note(ax, n)
        save(fig, "fig12_delta_K_avg.png")

    def fig13():
        xs, xe, ys, ye = [], [], [], []
        for d in DELTAS:
            runs = load_runs(d_name(d), S)
            cpu = [r["summary"]["cpu_time_s"] for r in runs]
            end = [movavg(r["rate_ratio"], 200)[-1] for r in runs]
            xs.append(np.mean(cpu)); xe.append(np.std(cpu))
            ys.append(np.mean(end)); ye.append(np.std(end))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(xs, ys, xerr=xe, yerr=ye, fmt="o-", color=COLORS[0], ecolor="#9dc8f7",
                    elinewidth=1.2, capsize=3, markersize=6)
        for k, (x, y, d) in enumerate(zip(xs, ys, DELTAS)):
            ax.annotate("Δ=%d" % d, (x, y), textcoords="offset points",
                        xytext=(6, -14) if k % 2 == 0 else (6, 8), fontsize=8, color=INK2)
        ax.set_xlabel("CPU time for 10,000 frames (s)")
        ax.set_ylabel("Normalized rate at the end\n(200-frame moving avg.)")
        ax.set_title("Fig. 13: rate vs. CPU time for different Δ (N = 10)")
        ax.margins(y=0.1)
        ax.text(0.99, 0.02, "mean of %d seeds, error bars = ±1 std" % len(runs),
                transform=ax.transAxes, ha="right", fontsize=7.5, color=INK2)
        save(fig, "fig13_delta_tradeoff_avg.png")

    def fig14_15():
        # baselines (CD reference, LR, Edge, Local) do not depend on the seed:
        # reuse the saved single runs of table2_fig14_15.py
        base = PROJECT_ROOT / "results" / "table2_fig14_15"
        means, sds, boxes, rows = {}, {}, {}, []
        n = 0
        for N_ in NS:
            runs = load_runs("table2_droo_N%d" % N_, S)
            n = len(runs)
            nt = len(runs[0]["rate_test"])
            ref = np.load(base / ("edge_N%d.npz" % N_))["reference"][:nt]
            per_seed = [r["rate_test"].mean() for r in runs]
            ratios = np.concatenate([r["rate_ratio"][-len(ref):] for r in runs])
            seed_ratio = [r["rate_ratio"][-len(ref):].mean() for r in runs]
            lr = np.load(base / ("lr_N%d.npz" % N_))
            means[N_] = {"CD": ref.mean(), "DROO": np.mean(per_seed),
                         "LR": lr["rate"].mean(),
                         "Edge Computing": np.load(base / ("edge_N%d.npz" % N_))["rate"].mean(),
                         "Local Computing": np.load(base / ("local_N%d.npz" % N_))["rate"].mean()}
            sds[N_] = np.std(per_seed)
            boxes[N_] = (ratios, lr["rate"] / lr["reference"])
            rows.append((N_, np.mean(seed_ratio), np.std(seed_ratio), np.min(seed_ratio),
                         np.max(seed_ratio), np.median(ratios), (lr["rate"] / lr["reference"]).mean()))
        methods = ["CD", "DROO", "LR", "Edge Computing", "Local Computing"]
        fig, ax = plt.subplots(figsize=(8.5, 5))
        w, x = 0.16, np.arange(len(NS))
        for k, m in enumerate(methods):
            vals = [means[N_][m] / 1e6 for N_ in NS]
            err = [sds[N_] / 1e6 for N_ in NS] if m == "DROO" else None
            ax.bar(x + (k - 2) * w, vals, w * 0.88, label=m, color=COLORS[k], yerr=err,
                   capsize=3, error_kw={"elinewidth": 1, "ecolor": INK})
        ax.set_xticks(x)
        ax.set_xticklabels(["N = %d" % N_ for N_ in NS])
        ax.set_ylabel("Average computation rate (Mbit/s)")
        ax.set_title("Fig. 14: computation rate of the offloading methods")
        ax.grid(axis="x", visible=False)
        ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
        ax.text(0.01, 0.98, "DROO: mean of %d seeds, error bar = ±1 std;\n"
                "other methods do not depend on the seed" % n,
                transform=ax.transAxes, va="top", fontsize=7.5, color=INK2)
        save(fig, "fig14_methods_avg.png")

        fig, ax = plt.subplots(figsize=(8, 5))
        data, pos, cols = [], [], []
        for k, N_ in enumerate(NS):
            data += list(boxes[N_])
            pos += [3 * k + 0.6, 3 * k + 1.4]
            cols += [COLORS[0], COLORS[3]]
        bp = ax.boxplot(data, positions=pos, widths=0.6, showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "linewidth": 1.5})
        for p, c in zip(bp["boxes"], cols):
            p.set_facecolor(c)
            p.set_alpha(0.55)
        ax.set_xticks([3 * k + 1 for k in range(len(NS))])
        ax.set_xticklabels(["N = %d" % N_ for N_ in NS])
        ax.set_ylabel("Normalized computation rate")
        ax.set_ylim(0.95, 1.005)
        ax.grid(axis="x", visible=False)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=COLORS[0], alpha=0.55, label="DROO (%d seeds pooled)" % n),
                           Patch(color=COLORS[3], alpha=0.55, label="LR")], loc="lower left")
        ax.set_title("Fig. 15: normalized rate on the test channels\n"
                     "box = 25th-75th percentile, line = median, whiskers = 1.5 x IQR")
        save(fig, "fig15_boxplot_avg.png")
        return rows

    for fn in (fig04, fig05, fig06, fig07, fig08, fig09, fig10, fig11, fig12, fig13):
        attempt(fn)
    rows = None
    try:
        rows = fig14_15()
    except FileNotFoundError as e:
        print("skipped fig14_15:", e)
    write_summary(S, rows)


def write_summary(seeds, table_rows):
    """Average normalized rate on the last 2,000 frames, mean +/- std over seeds."""
    lines = ["# Averaged results (seeds %d-%d)" % (min(seeds), max(seeds)), "",
             "Average normalized rate over the test frames (the last 20% of each run), "
             "for every setting: mean, standard deviation, lowest and highest over the seeds.", "",
             "| Setting | Seeds | Mean | Std | Min | Max |", "|---|---|---|---|---|---|"]
    for name in SETTINGS:
        try:
            runs = load_runs(name, seeds)
        except FileNotFoundError:
            continue
        v = [r["summary"]["avg_normalized_rate_test"] for r in runs]
        lines.append("| %s | %d | %.5f | %.5f | %.5f | %.5f |"
                     % (name, len(v), np.mean(v), np.std(v), np.min(v), np.max(v)))
    if table_rows:
        lines += ["", "# Fig. 15 numbers (DROO over seeds, test channels)", "",
                  "| N | DROO mean | DROO std over seeds | worst seed | best seed | DROO median (pooled) | LR mean |",
                  "|---|---|---|---|---|---|---|"]
        for r in table_rows:
            lines.append("| %d | %.5f | %.5f | %.5f | %.5f | %.5f | %.5f |" % r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("Saved", OUT / "summary.md")


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Multi-seed averaged figures")
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    p.add_argument("--only", nargs="*", help="groups: " + ", ".join(GROUPS))
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    seeds = range(a.seeds)

    names = list(SETTINGS) if not a.only else \
        list(dict.fromkeys(n for g in a.only for n in GROUPS[g]))
    # longest first: 30,000-frame runs, then bisection runs, then the rest
    names.sort(key=lambda n: (not n.startswith("table2"), not n.startswith("fig11")))
    jobs = [(n, s) for s in seeds for n in names]
    jobs.sort(key=lambda j: (not j[0].startswith("table2"), not j[0].startswith("fig11")))
    todo = [j for j in jobs if not run_path(*j).exists()]
    print("%d runs, %d done, %d to run, %d at a time" % (len(jobs), len(jobs) - len(todo),
                                                         len(todo), a.workers), flush=True)
    if a.dry_run:
        for n, s in todo:
            print("  %-26s seed %d" % (n, s))
        return
    if not a.plot_only and todo:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(run_job, n, s): (n, s) for n, s in todo}
            for k, f in enumerate(as_completed(futs), 1):
                n, s = futs[f]
                try:
                    secs = f.result()
                    status = "%5.0f s" % secs
                except Exception as e:      # keep going; report at the end
                    status = "FAILED: %r" % e
                print("[%3d/%d] %-26s seed %d  %s   (elapsed %.1f min)"
                      % (k, len(todo), n, s, status, (time.time() - t0) / 60), flush=True)
    plot_all(seeds)


if __name__ == "__main__":
    main()
