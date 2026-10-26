"""
fig07.py - reproduces Fig. 7 of the paper (Section 5.1, temporarily
critical computation demand). NOT in the authors' repository.

Paper setup: N = 10, K = 10, 10,000 frames. Default weights are
[1, 1.5, 1, 1.5, ...], so WD1 has weight 1 and WD2 has weight 1.5.
    t = 4,000 : WD2's weight doubled   1.5 -> 3
    t = 6,000 : WD1's weight tripled   1   -> 3
    t = 8,000 : both reset to the original values

What we compute:
  - Top panel: each WD's "relative computation rate" = its rate under
    DROO's action divided by its rate under the optimal action for the
    ORIGINAL weights (the authors' stored output_mode for data_10).
  - Bottom panel: the normalized rate Q-hat. While weights are changed, the
    stored optimum no longer applies, so we compute the true optimum by
    trying all 1,024 actions (baselines/p2_vectorized.py, checked against
    the authors' solver). This is cached in results/fig07/reference.npz.
  - Data order as in the authors' demos: frame i uses channel sample i.

Paper's claims to check:
  - first ~3,000 frames: both WDs' relative rates are below 1 most of the time
  - 4,000 < t < 8,000: WD2's rate rises clearly (up to 2-3x at times);
    similar for WD1 during 6,000 < t < 8,000; back to ~1 after 8,000
  - on average WD1 and WD2 get 26% and 12% higher rates in their
    increased-weight periods
  - DROO quickly adapts: normalized rate stays high

How to run (from the project folder):
    python experiments/fig07.py --run main
    python experiments/fig07.py --plot
"""

import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from common import (FIGURES_DIR, frames, PROJECT_ROOT, RESULTS_DIR, load_data,
                    load_result, rolling, run_droo, save_result)

sys.path.insert(0, str(PROJECT_ROOT / "baselines"))
from p2_vectorized import all_actions, optimum, solve_p2  # noqa: E402

EXPERIMENT = "fig07"
N, SEED = 10, 0
FRAMES = frames(10000)
JOBS = ["main"]
# 4,000 / 6,000 / 8,000 for 10,000 frames
T_WD2, T_WD1, T_RESET = int(.4 * FRAMES), int(.6 * FRAMES), int(.8 * FRAMES)

W_DEFAULT = np.array([1, 1.5] * (N // 2), dtype=float)
W_A = W_DEFAULT.copy()
W_A[1] = 3.0                      # WD2 doubled
W_B = W_A.copy()
W_B[0] = 3.0                      # WD1 tripled as well


def weights_at(i):
    if T_WD2 <= i < T_WD1:
        return W_A
    if T_WD1 <= i < T_RESET:
        return W_B
    return W_DEFAULT


def job_tag(job):
    return "seed%d" % SEED


def reference_rates():
    """True optimum for every frame under the weights active at that frame."""
    path = RESULTS_DIR / EXPERIMENT / "reference.npz"
    if path.exists():
        return np.load(path)["rate"]
    channel, rate = load_data("data_10")
    h_real = channel / 1000000
    ref = rate[:, 0].copy()               # stored optimum = default weights
    actions = all_actions(N)
    print("Computing the true optimum for frames %d-%d (all 1,024 actions)..."
          % (T_WD2, T_RESET - 1), flush=True)
    for i in range(T_WD2, T_RESET):
        ref[i] = optimum(h_real[i], weights_at(i), actions)[0]
        if (i - T_WD2) % 500 == 0:
            print("  reference: frame %d" % i, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rate=ref)
    return ref


def run(job):
    ref = reference_rates()
    rate_col = ref[:, None]                # same shape as the stored rates

    def change_weights(i, state):
        if i == 0:
            state["weights"], state["rate"] = list(W_DEFAULT), rate_col
        if i in (T_WD2, T_WD1, T_RESET):
            state["weights"] = list(weights_at(i))
            print("  frame %d: weights %s" % (i, state["weights"]))

    print("Fig. 7: N=10, K=10, %d frames, seed %d" % (FRAMES, SEED))
    result = run_droo(N=N, n=FRAMES, K=N, delta=None, seed=SEED,
                      hook=change_weights, split=False)

    # Per-WD rates: DROO's action vs. the optimal action for original weights
    channel, _ = load_data("data_10")
    h_real = channel / 1000000
    from scipy.io import loadmat
    best_mode = loadmat(str(PROJECT_ROOT / "DROO" / "data" / "data_10"))["output_mode"]
    rel = np.zeros((FRAMES, 2))
    for i in range(FRAMES):
        droo_wd = solve_p2(h_real[i], result["mode"][i], weights_at(i))["r_wd"][0]
        base_wd = solve_p2(h_real[i], best_mode[i], W_DEFAULT)["r_wd"][0]
        rel[i] = droo_wd[:2] / base_wd[:2]
    result["relative_wd12"] = rel

    s = result["summary"]
    s["wd1_gain_in_tripled_period"] = float(rel[T_WD1:T_RESET, 0].mean() - 1)
    s["wd2_gain_in_doubled_period"] = float(rel[T_WD2:T_RESET, 1].mean() - 1)
    s["wd1_relative_first_3000"] = float(rel[:int(.3 * FRAMES), 0].mean())
    s["wd2_relative_first_3000"] = float(rel[:int(.3 * FRAMES), 1].mean())
    s["wd2_max_relative_4000_8000"] = float(rel[T_WD2:T_RESET, 1].max())
    mean50, _, _ = rolling(result["rate_ratio"], 50)
    s["min_moving_avg_after_4000"] = float(mean50[T_WD2:].min())
    save_result(EXPERIMENT, job_tag(job), result)
    np.save(RESULTS_DIR / EXPERIMENT / (job_tag(job) + "_relative_wd12.npy"), rel)

    print()
    print("WD2 average gain while its weight is doubled (4,000-8,000): %+.1f%% (paper: +12%%)"
          % (100 * s["wd2_gain_in_doubled_period"]))
    print("WD1 average gain while its weight is tripled (6,000-8,000): %+.1f%% (paper: +26%%)"
          % (100 * s["wd1_gain_in_tripled_period"]))
    print("Lowest moving average of normalized rate after 4,000: %.4f"
          % s["min_moving_avg_after_4000"])


def plot():
    r = load_result(EXPERIMENT, "seed%d" % SEED)
    rel = np.load(RESULTS_DIR / EXPERIMENT / ("seed%d_relative_wd12.npy" % SEED))
    t = np.arange(1, FRAMES + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for k, (name, color) in enumerate([("WD1", "tab:blue"), ("WD2", "tab:red")]):
        mean, _, _ = rolling(rel[:, k], 50)
        ax1.plot(t, mean, color=color, linewidth=1, label=name + " (50-frame mean)")
    ax1.axhline(1, color="gray", linewidth=0.8)
    ax1.set_ylabel("Relative computation rate")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    mean, lo, hi = rolling(r["rate_ratio"], 50)
    ax2.fill_between(t, lo, hi, color="tab:blue", alpha=0.2, linewidth=0)
    ax2.plot(t, mean, color="tab:blue", linewidth=1.2)
    ax2.set_ylim(0.8, 1.0)
    ax2.set_ylabel("Normalized computation rate")
    ax2.set_xlabel("Time frame t")
    ax2.grid(alpha=0.3)
    for frame, label in [(T_WD2, "WD2 x2"), (T_WD1, "WD1 x3"), (T_RESET, "reset")]:
        for ax in (ax1, ax2):
            ax.axvline(frame, color="gray", linestyle="--", linewidth=0.8)
        ax1.annotate(label, (frame, ax1.get_ylim()[1]), fontsize=8, ha="center",
                     va="bottom", annotation_clip=False)
    fig.suptitle("Fig. 7 replication: temporarily new weights (N=10, K=10)")
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "fig07_new_weights.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Fig. 7")
    parser.add_argument("--run", help="the job: main")
    parser.add_argument("--plot", action="store_true", help="draw the figure")
    args = parser.parse_args()
    if args.run:
        run(args.run)
    if args.plot:
        plot()
