"""
fig04.py - reproduces Fig. 4 of the paper (Section 4.4).

Fig. 4 shows k*_t, the rank (1 = first candidate) of the best offloading
action among the K = N = 10 candidates, for N = 10, over 10,000 frames:
50-frame moving average plus the min-max band. It motivates adaptive K.
This is exactly the Fig. 5 setting, so it is drawn from the saved Fig. 5
results (results/fig05/seed0.npz); no new run is needed.

Paper's claim to check:
  - the selected index k*_t is no larger than 5 once t >= 5,000

How to run (from the project folder, after fig05.py):
    python experiments/fig04.py --plot
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES_DIR, load_result, rolling

JOBS = []          # nothing to run; uses fig05's results


def plot():
    try:
        r = load_result("fig05", "seed0")
    except FileNotFoundError:
        print("Fig. 4 needs results/fig05/seed0 - run experiments/fig05.py first.")
        return
    k_star = r["k_idx"] + 1                 # code counts from 0, paper from 1
    t = np.arange(1, len(k_star) + 1)
    mean, lo, hi = rolling(k_star, 50)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(t, lo, hi, color="tab:purple", alpha=0.25, linewidth=0)
    ax.plot(t, mean, color="tab:blue", linewidth=1)
    ax.set_ylim(0.5, 10.5)
    ax.set_xlabel("Time frame t")
    ax.set_ylabel("Selected index k*_t")
    ax.set_title("Fig. 4 replication: rank of the best action (N=10, K=10)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / "fig04_kstar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    late = k_star[5000:]
    print("After t = 5,000: largest k* = %d; share of frames with k* <= 5: %.2f%%"
          % (late.max(), 100 * (late <= 5).mean()))
    print("Saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Fig. 4")
    parser.add_argument("--plot", action="store_true", help="draw the figure")
    args = parser.parse_args()
    if args.plot:
        plot()
