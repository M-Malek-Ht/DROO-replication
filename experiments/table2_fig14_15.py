"""
table2_fig14_15.py - reproduces Table 2, Fig. 14 and Fig. 15 of the paper
(Sections 5.3 and 5.4) for N = 10, 20 and 30 WDs.

Paper setup:
  - DROO is trained on 24,000 channels and then evaluated on 6,000 new ones
    (the authors' 80/20 split of 30,000 frames, as in mainPyTorch.py);
    adaptive K with Delta = 32 ("set Delta = 32 in the following simulations")
  - Fig. 14: average weighted sum computation rate on the 6,000 test
    channels for CD, DROO, LR, Edge Computing and Local Computing
  - Fig. 15: box plot of DROO's and LR's normalized rate. Normalized against
    the true optimum for N = 10 and against CD for N = 20, 30. The authors'
    data files already store exactly these reference values (README of
    DROO/data: exhaustive search for N <= 10, CD for N = 20, 30).
  - Table 2: CPU time per channel for DROO (fixed K = N), DROO (adaptive K,
    Delta = 32), CD and LR

Jobs (each one is a separate process, so run_all.py can run them in parallel):
  droo_adaptive_N<N>   30,000 frames, adaptive K (Delta = 32): rates + time
  droo_fixed_N<N>      fixed K = N, timing only. 5,000 frames instead of
                       30,000 to save time (with a fixed K the cost per frame
                       does not shrink over time, unlike adaptive K)
  edge_N<N>, local_N<N>, lr_N<N>   the baselines on the same 6,000 test
                       channels as DROO
  cd_N<N>              the authors' CD method on the first 100 test channels
                       (timing; CD's rates are the stored reference values)

Timing note: run_all.py runs jobs side by side, so we report CPU time
(time.process_time), which counts only each process's own work. Absolute
times also depend on the computer (paper: Intel i5-4570); the comparison
that should carry over is the ratio between methods.

How to run (from the project folder):
    python experiments/table2_fig14_15.py --run droo_adaptive_N10
    python experiments/table2_fig14_15.py --plot
"""

import argparse
import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from common import (FIGURES_DIR, frames, PROJECT_ROOT, RESULTS_DIR, environment_info,
                    load_data, load_result, run_droo, save_result)

sys.path.insert(0, str(PROJECT_ROOT / "baselines"))
from methods import (LinearRelaxation, coordinate_descent,  # noqa: E402
                     edge_computing, local_computing)

EXPERIMENT = "table2_fig14_15"
NS = [10, 20, 30]
SEED = 0
TRAIN_FRAMES = frames(30000, 500)
FIXED_K_FRAMES = frames(5000, 200)
CD_CHANNELS = frames(100, 3)
LR_FRESH_CHANNELS = frames(30, 3)
# Long jobs first, so run_all.py starts them early
JOBS = (["droo_adaptive_N%d" % n for n in (30, 20, 10)]
        + ["droo_fixed_N%d" % n for n in (30, 20, 10)]
        + ["cd_N%d" % n for n in (30, 20, 10)]
        + ["%s_N%d" % (m, n) for m in ("lr", "edge", "local") for n in NS])


def job_tag(job):
    return job


def test_indices(N):
    """The 6,000 test channels DROO is evaluated on (same as mainPyTorch.py)."""
    channel, _ = load_data("data_%d" % N)
    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, TRAIN_FRAMES - int(.8 * TRAIN_FRAMES))
    return np.arange(split_idx, split_idx + num_test)


def save_baseline(job, rates, reference, cpu_time, count, extra=None):
    folder = RESULTS_DIR / EXPERIMENT
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (job + ".npz"), rate=np.asarray(rates),
                        reference=np.asarray(reference))
    summary = {"channels": count,
               "avg_normalized_rate": float(np.mean(np.asarray(rates) / reference)),
               "cpu_time_s": cpu_time,
               "cpu_time_per_channel_s": cpu_time / count}
    summary.update(extra or {})
    with open(folder / (job + "_summary.json"), "w") as f:
        json.dump({"summary": summary, "environment": environment_info()}, f, indent=2)
    print("%s: avg normalized rate %.5f, CPU %.4f s per channel"
          % (job, summary["avg_normalized_rate"], summary["cpu_time_per_channel_s"]))


def run(job):
    method, n_part = job.rsplit("_N", 1)
    N = int(n_part)
    channel, rate = load_data("data_%d" % N)
    h_real = channel / 1000000          # load_data multiplies by 1e6
    reference = rate[:, 0]
    idx = test_indices(N)

    if method == "droo_adaptive":
        result = run_droo(N=N, n=TRAIN_FRAMES, K=N, delta=32, seed=SEED)
        save_result(EXPERIMENT, job, result)
        print("avg normalized rate %.5f, CPU %.4f s per frame"
              % (result["summary"]["avg_normalized_rate_test"],
                 result["summary"]["cpu_time_per_frame_s"]))
        return
    if method == "droo_fixed":
        result = run_droo(N=N, n=FIXED_K_FRAMES, K=N, delta=None, seed=SEED)
        save_result(EXPERIMENT, job, result)
        print("CPU %.4f s per frame" % result["summary"]["cpu_time_per_frame_s"])
        return

    np.random.seed(SEED)               # CD starts from a random action
    if method == "cd":
        idx = idx[:CD_CHANNELS]
        solver = coordinate_descent
    elif method == "lr":
        solver = LinearRelaxation(N)   # built once, re-solved per channel
    elif method == "edge":
        solver = edge_computing
    elif method == "local":
        solver = local_computing
    else:
        raise ValueError(job)

    start = time.process_time()
    rates = [solver(h_real[i])[0] for i in idx]
    cpu = time.process_time() - start

    extra = {}
    if method == "lr":
        # The paper's LR time (0.24 s for N=10) suggests the convex problem
        # was set up from scratch for every channel. Measure that too.
        start = time.process_time()
        for i in idx[:LR_FRESH_CHANNELS]:
            LinearRelaxation(N)(h_real[i])
        extra["cpu_time_per_channel_rebuilt_s"] = \
            (time.process_time() - start) / LR_FRESH_CHANNELS
    save_baseline(job, rates, reference[idx], cpu, len(idx), extra)


# ---------------------------------------------------------------------------
# Plots and the table
# ---------------------------------------------------------------------------
def load_baseline(job):
    folder = RESULTS_DIR / EXPERIMENT
    data = dict(np.load(folder / (job + ".npz")))
    with open(folder / (job + "_summary.json")) as f:
        data.update(json.load(f))
    return data


def plot():
    table, means, boxes = [], {}, {}
    for N in NS:
        ref = load_data("data_%d" % N)[1][:, 0][test_indices(N)]
        ad = load_result(EXPERIMENT, "droo_adaptive_N%d" % N)
        fx = load_result(EXPERIMENT, "droo_fixed_N%d" % N)
        cd = load_baseline("cd_N%d" % N)
        lr = load_baseline("lr_N%d" % N)
        edge = load_baseline("edge_N%d" % N)
        local = load_baseline("local_N%d" % N)

        droo_rate = ad["rate"][-len(ref):]          # the 6,000 test frames
        droo_ratio = ad["rate_ratio"][-len(ref):]
        lr_ratio = lr["rate"] / lr["reference"]

        means[N] = {"CD": ref.mean(), "DROO": droo_rate.mean(),
                    "LR": lr["rate"].mean(), "Edge Computing": edge["rate"].mean(),
                    "Local Computing": local["rate"].mean()}
        boxes[N] = (droo_ratio, lr_ratio)
        table.append({
            "N": N,
            "droo_fixed": fx["summary"]["cpu_time_per_frame_s"],
            "droo_adaptive": ad["summary"]["cpu_time_per_frame_s"],
            "cd": cd["summary"]["cpu_time_per_channel_s"],
            "cd_ratio": cd["summary"]["avg_normalized_rate"],
            "lr": lr["summary"]["cpu_time_per_channel_s"],
            "lr_rebuilt": lr["summary"]["cpu_time_per_channel_rebuilt_s"],
            "droo_median": float(np.median(droo_ratio)),
            "lr_median": float(np.median(lr_ratio)),
            "droo_mean": float(droo_ratio.mean()),
            "lr_mean": float(lr_ratio.mean()),
            "droo_q25": float(np.percentile(droo_ratio, 25)),
            "lr_q25": float(np.percentile(lr_ratio, 25)),
            # share of channels where DROO beats the reference by more than
            # rounding (stored and recomputed optima differ by ~1e-7)
            "droo_above_1": float((droo_ratio > 1 + 1e-5).mean()),
            "droo_above_099": float((droo_ratio >= 0.99).mean()),
            "lr_above_099": float((lr_ratio >= 0.99).mean()),
            "droo_p5": float(np.percentile(droo_ratio, 5)),
        })

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Fig. 14: bar chart -------------------------------------------------
    methods = ["CD", "DROO", "LR", "Edge Computing", "Local Computing"]
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.16
    x = np.arange(len(NS))
    for k, m in enumerate(methods):
        ax.bar(x + (k - 2) * width, [means[N][m] / 1e6 for N in NS], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([str(N) for N in NS])
    ax.set_xlabel("Number of WDs N")
    ax.set_ylabel("Average computation rate Q (x 1e6 bits/s)")
    ax.set_title("Fig. 14 replication: computation rate of offloading algorithms")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig14_methods.png", dpi=150)
    plt.close(fig)

    # ---- Fig. 15: box plot --------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    data, pos, colors = [], [], []
    for k, N in enumerate(NS):
        data += [boxes[N][0], boxes[N][1]]
        pos += [3 * k + 0.6, 3 * k + 1.4]
        colors += ["lightblue", "khaki"]
    bp = ax.boxplot(data, positions=pos, widths=0.6, showfliers=False,
                    whis=1.5, patch_artist=True,
                    medianprops={"color": "red"})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_xticks([3 * k + 1 for k in range(len(NS))])
    ax.set_xticklabels(["N = %d" % N for N in NS])
    ax.set_ylabel("Normalized computation rate")
    ax.set_ylim(0.95, 1.005)
    ax.set_title("Fig. 15 replication: DROO (blue) vs LR (yellow)\n"
                 "box = 25th-75th percentile, red = median, whiskers = 1.5 x IQR")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig15_boxplot.png", dpi=150)
    plt.close(fig)

    # ---- Table 2 + numbers for the report -----------------------------------
    paper = {10: ("3.6e-2", "1.2e-2", "2.0e-1", "2.4e-1"),
             20: ("1.3e-1", "3.0e-2", "1.3", "5.3e-1"),
             30: ("3.1e-1", "5.9e-2", "3.8", "8.1e-1")}
    lines = [
        "# Table 2 replication: CPU execution latency per channel (seconds)",
        "",
        "Ours = CPU time on this computer; paper = Intel i5-4570 (in brackets).",
        "",
        "| N | DROO fixed K=N | DROO adaptive (Delta=32) | CD | LR (solver reused) | LR (rebuilt per channel) |",
        "|---|---|---|---|---|---|",
    ]
    for row in table:
        p = paper[row["N"]]
        lines.append("| %d | %.1e (%s) | %.1e (%s) | %.1e (%s) | %.1e (%s) | %.1e |" % (
            row["N"], row["droo_fixed"], p[0], row["droo_adaptive"], p[1],
            row["cd"], p[2], row["lr"], p[3], row["lr_rebuilt"]))
    # The paper's claims are about ratios ("CD and LR take 65 and 14 times
    # longer" for N = 30), which hold up better than absolute times when
    # several runs share the computer.
    lines += ["", "How many times slower than DROO (adaptive K) - ours (paper):", "",
              "| N | DROO fixed K / adaptive | CD / DROO | LR reused / DROO | LR rebuilt / DROO |",
              "|---|---|---|---|---|"]
    for row in table:
        p = [float(v) for v in paper[row["N"]]]
        d = row["droo_adaptive"]
        lines.append("| %d | %.1fx (%.1fx) | %.0fx (%.0fx) | %.1fx (%.0fx) | %.1fx (%.0fx) |" % (
            row["N"], row["droo_fixed"] / d, p[0] / p[1], row["cd"] / d, p[2] / p[1],
            row["lr"] / d, p[3] / p[1], row["lr_rebuilt"] / d, p[3] / p[1]))
    lines += ["", "# Fig. 14 numbers: average computation rate (bits/s) on the 6,000 test channels", "",
              "| N | CD | DROO | LR | Edge | Local |", "|---|---|---|---|---|---|"]
    for N in NS:
        m = means[N]
        lines.append("| %d | %.4g | %.4g | %.4g | %.4g | %.4g |" % (
            N, m["CD"], m["DROO"], m["LR"], m["Edge Computing"], m["Local Computing"]))
    lines += ["", "# Fig. 15 numbers: normalized rate on the 6,000 test channels", "",
              "| N | DROO median | DROO mean | DROO 25th pct | DROO 5th pct | DROO >= 0.99 | DROO > 1 | LR median | LR mean | LR 25th pct | LR >= 0.99 | CD (100 ch.) mean |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in table:
        lines.append("| %d | %.4f | %.4f | %.4f | %.4f | %.1f%% | %.1f%% | %.4f | %.4f | %.4f | %.1f%% | %.4f |" % (
            row["N"], row["droo_median"], row["droo_mean"], row["droo_q25"],
            row["droo_p5"], 100 * row["droo_above_099"], 100 * row["droo_above_1"],
            row["lr_median"], row["lr_mean"], row["lr_q25"],
            100 * row["lr_above_099"], row["cd_ratio"]))
    text = "\n".join(lines) + "\n"
    (RESULTS_DIR / EXPERIMENT / "table2_fig14_15.md").write_text(text)
    print(text)
    print("Saved", FIGURES_DIR / "fig14_methods.png")
    print("Saved", FIGURES_DIR / "fig15_boxplot.png")
    print("Saved", RESULTS_DIR / EXPERIMENT / "table2_fig14_15.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 2, Figs. 14-15")
    parser.add_argument("--run", help="one job, e.g. droo_adaptive_N10 or lr_N20")
    parser.add_argument("--plot", action="store_true", help="draw figures + table")
    args = parser.parse_args()
    if args.run:
        run(args.run)
    if args.plot:
        plot()
