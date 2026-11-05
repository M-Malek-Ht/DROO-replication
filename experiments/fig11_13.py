"""
fig11_13.py - reproduces Figs. 11, 12 and 13 of the paper
(Section 5.2, "Impact of Updating Intervals Delta").

DROO with adaptive K (paper Sec. 4.4) for N = 10 and different updating
intervals Delta. Delta = infinity means K never changes, i.e. fixed K = N.
One set of runs gives all three figures:
  Fig. 11: moving average (window 200) of the normalized rate,
           Delta = 1, 2, 4, 16, 128, 1024, infinity
  Fig. 12: moving average of K_t, same Deltas
  Fig. 13: average normalized rate at the end vs. total CPU time of the
           10,000 frames, Delta = 1, 2, 4, 8, 16, 32, ..., 1024

Paper's claims to check:
  - Fig. 11: DROO reaches the optimum only for a large enough Delta (>= 16)
  - Fig. 12: K_t is larger for larger Delta
  - Fig. 13: the rate rises quickly up to Delta = 16 (0.96 -> ~1), then
             only marginally; the CPU time keeps increasing with Delta;
             the paper picks Delta = 32 as the trade-off

Note on timing: run_all.py runs several experiments at the same time, so we
report CPU time (time.process_time), which counts only this run's own work.

How to run (from the project folder):
    python experiments/fig11_13.py --run 16      # one Delta
    python experiments/fig11_13.py --run inf
    python experiments/fig11_13.py --plot
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import (frames, FIGURES_DIR, load_result, moving_average_plot, rolling,
                    run_droo, save_result)

EXPERIMENT = "fig11_13"
N, SEED = 10, 0
FRAMES = frames(10000)
DELTAS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
JOBS = [str(d) for d in DELTAS] + ["inf"]
FIG11_12 = ["1", "2", "4", "16", "128", "1024", "inf"]


def job_tag(job):
    return "delta_" + job


def run(job):
    delta = None if job == "inf" else int(job)
    print("Fig. 11-13 run: Delta =", job)
    result = run_droo(N=N, n=FRAMES, K=N, delta=delta, seed=SEED)
    save_result(EXPERIMENT, job_tag(job), result)
    s = result["summary"]
    print("Average normalized rate (test frames): %.5f   CPU time %.0f s"
          % (s["avg_normalized_rate_test"], s["cpu_time_s"]))


def label(job):
    return "Delta = infinity (fixed K = N)" if job == "inf" else "Delta = " + job


def plot():
    runs = {job: load_result(EXPERIMENT, job_tag(job)) for job in JOBS}

    # Delta = 1024 gives exactly the same decisions as Delta = infinity in this
    # run (the winner was never among the candidates it trims), so its curve
    # lies on top of it; draw it dashed so both stay visible.
    # draw Delta = 1024 last, so its dashes sit on top of the infinity line
    order = [j for j in FIG11_12 if j != "1024"] + ["1024"]
    p = moving_average_plot([(label(j), runs[j]["rate_ratio"],
                              "--" if j == "1024" else "-") for j in order],
                            FIGURES_DIR / "fig11_delta_rate.png",
                            title="Fig. 11 replication: adaptive K, different Delta (N=10)",
                            ylim=(0.90, 1.0))
    print("Saved", p)

    p = moving_average_plot([(label(j), runs[j]["K"]) for j in FIG11_12],
                            FIGURES_DIR / "fig12_delta_K.png",
                            title="Fig. 12 replication: K_t under different Delta (N=10)",
                            ylabel="Moving average of K_t", ylim=(1, 10.5))
    print("Saved", p)

    # Fig. 13: final moving average vs. CPU time
    xs, ys = [], []
    for d in DELTAS:
        r = runs[str(d)]
        mean200, _, _ = rolling(r["rate_ratio"], 200)
        xs.append(r["summary"]["cpu_time_s"])
        ys.append(mean200[-1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, "o-", color="tab:blue")
    for k, (x, y, d) in enumerate(zip(xs, ys, DELTAS)):
        # alternate below/above so neighbouring labels do not overlap
        offset = (4, -14) if k % 2 == 0 else (4, 8)
        ax.annotate("%d" % d, (x, y), textcoords="offset points",
                    xytext=offset, fontsize=8)
    ax.text(0.98, 0.04, "labels = Delta", transform=ax.transAxes,
            ha="right", fontsize=8, color="gray")
    ax.margins(y=0.08)
    ax.set_xlabel("CPU execution latency for 10,000 frames (seconds)")
    ax.set_ylabel("Moving average of normalized rate at the end")
    ax.set_title("Fig. 13 replication: rate vs. CPU latency (N=10)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIGURES_DIR / "fig13_delta_tradeoff.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("Saved", p)

    # Also print the Fig. 13 numbers as a small table
    print("\nDelta | final moving avg | CPU time (s)")
    for x, y, d in zip(xs, ys, DELTAS):
        print("%5d | %.5f          | %.1f" % (d, y, x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Figs. 11-13")
    parser.add_argument("--run", help="one Delta, e.g. 16, or inf")
    parser.add_argument("--plot", action="store_true", help="draw the figures")
    args = parser.parse_args()
    if args.run:
        run(args.run)
    if args.plot:
        plot()
