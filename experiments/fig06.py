"""
fig06.py - reproduces Fig. 6 of the paper (Section 5.1, alternating weights).

Paper setup: N = 10 WDs, K = 10, 10,000 time frames. At frames 6,000 and
8,000 the weights of ALL WDs are swapped at once:
    default  : [1, 1.5, 1, 1.5, ...]  (odd WDs 1, even WDs 1.5)
    swapped  : [1.5, 1, 1.5, 1, ...]
The "best known" rate for the swapped weights comes from the authors' file
DROO/data/data_10_WeightsAlternated.mat (same channels, re-solved).

Based on DROO/demo_alternate_weights.py (same schedule, same data order:
frame i uses sample i). One choice to note: the paper's caption says K = 10,
while the authors' demo uses adaptive K (Delta = 32). We follow the paper by
default; add --adaptive to run the demo's setting instead.

Paper's claims to check:
  - the training loss jumps after each swap, then settles again within
    about 1,000 frames
  - for t > 6,000: the minimum normalized rate stays above 0.95 and the
    50-frame moving average stays above 0.99

How to run (from the project folder):
    python experiments/fig06.py               # paper setting (fixed K = 10)
    python experiments/fig06.py --adaptive    # authors' demo setting
    python experiments/fig06.py --frames 500  # quick test

Outputs:
    results/fig06/<tag>.npz and <tag>_summary.json
    figures/fig06_<tag>.png
"""

import argparse

from common import (FIGURES_DIR, load_data, plot_loss_and_rate, rolling,
                    run_droo, save_result)

parser = argparse.ArgumentParser(description="Reproduce Fig. 6")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--frames", type=int, default=10000, help="time frames")
parser.add_argument("--adaptive", action="store_true",
                    help="use adaptive K (Delta=32) like the authors' demo")
args = parser.parse_args()

N = 10
n = args.frames
delta = 32 if args.adaptive else None

DEFAULT_WEIGHTS = [1, 1.5] * (N // 2)
SWAPPED_WEIGHTS = [1.5, 1] * (N // 2)
_, RATE_DEFAULT = load_data("data_10")
_, RATE_SWAPPED = load_data("data_10_WeightsAlternated")

SWAP_FRAMES = [int(0.6 * n), int(0.8 * n)]   # 6,000 and 8,000 for n = 10,000


def swap_weights(i, state):
    """Called at the start of every frame; swaps weights at the swap frames."""
    if i == 0:
        state["weights"], state["rate"] = DEFAULT_WEIGHTS, RATE_DEFAULT
    elif i == SWAP_FRAMES[0]:
        state["weights"], state["rate"] = SWAPPED_WEIGHTS, RATE_SWAPPED
        print("  frame %d: weights swapped to %s" % (i, SWAPPED_WEIGHTS))
    elif i == SWAP_FRAMES[1]:
        state["weights"], state["rate"] = DEFAULT_WEIGHTS, RATE_DEFAULT
        print("  frame %d: weights reset to %s" % (i, DEFAULT_WEIGHTS))


print("Fig. 6: N=%d, K=%s, %d frames, seed %d"
      % (N, "adaptive (Delta=32)" if args.adaptive else "10 (fixed)", n, args.seed))

result = run_droo(N=N, n=n, K=N, delta=delta, seed=args.seed,
                  hook=swap_weights, split=False)

# Paper's specific claims about the part after the first swap
after = result["rate_ratio"][SWAP_FRAMES[0]:]
mean50, _, _ = rolling(result["rate_ratio"], 50)
result["summary"]["min_rate_after_first_swap"] = float(after.min())
result["summary"]["min_moving_avg_after_first_swap"] = float(mean50[SWAP_FRAMES[0]:].min())
result["summary"]["swap_frames"] = SWAP_FRAMES

tag = "seed%d" % args.seed
if args.adaptive:
    tag += "_adaptive"
if n != 10000:
    tag += "_frames%d" % n

folder = save_result("fig06", tag, result)
figure = plot_loss_and_rate(
    result["loss"], result["rate_ratio"], train_interval=10,
    path=FIGURES_DIR / ("fig06_%s.png" % tag),
    title="Fig. 6 replication: alternating weights (N=10, K=%s, seed %d)"
          % ("adaptive" if args.adaptive else "10", args.seed),
    events=[(SWAP_FRAMES[0], "swap"), (SWAP_FRAMES[1], "reset")])

s = result["summary"]
print()
print("Average normalized rate (last %d frames): %.5f"
      % (result["config"]["num_test"], s["avg_normalized_rate_test"]))
print("After frame %d: lowest single rate %.4f (paper: > 0.95)"
      % (SWAP_FRAMES[0], s["min_rate_after_first_swap"]))
print("After frame %d: lowest moving average %.4f (paper: > 0.99)"
      % (SWAP_FRAMES[0], s["min_moving_avg_after_first_swap"]))
print("Total time: %.1f s   |   per frame: %.4f s"
      % (s["total_time_s"], s["time_per_frame_s"]))
print("Saved numbers to:", folder)
print("Saved graph to:  ", figure)
