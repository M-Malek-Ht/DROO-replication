"""
fig05.py - reproduces Fig. 5 of the paper (Section 5.1, "Convergence Performance").

Paper setup: N = 10 WDs, fixed K = N (no adaptive K), 10,000 time frames,
memory 1024, batch 128, training interval 10, learning rate 0.01.

Paper's claims to check:
  - the moving average of the normalized rate exceeds 0.98 early (t > 400)
    and approaches 1 as t grows
  - the training loss decreases and settles at around 0.04

How to run (from the project folder):
    python experiments/fig05.py              # full run
    python experiments/fig05.py --frames 500 # quick test (a few seconds)

Outputs:
    results/fig05/seed<S>.npz and seed<S>_summary.json
    figures/fig05_seed<S>.png
"""

import argparse

from common import FIGURES_DIR, plot_loss_and_rate, run_droo, save_result

parser = argparse.ArgumentParser(description="Reproduce Fig. 5")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--frames", type=int, default=10000, help="time frames")
args = parser.parse_args()

N = 10
print("Fig. 5: N=%d, fixed K=N, %d frames, seed %d" % (N, args.frames, args.seed))

result = run_droo(N=N, n=args.frames, K=N, delta=None, seed=args.seed)

tag = "seed%d" % args.seed
if args.frames != 10000:
    tag += "_frames%d" % args.frames   # keep quick tests apart from real runs

folder = save_result("fig05", tag, result)
figure = plot_loss_and_rate(
    result["loss"], result["rate_ratio"], train_interval=10,
    path=FIGURES_DIR / ("fig05_%s.png" % tag),
    title="Fig. 5 replication (N=10, K=N, seed %d)" % args.seed)

s = result["summary"]
print()
print("Average normalized rate (test frames): %.5f" % s["avg_normalized_rate_test"])
print("Final training loss (mean of last 50): %.4f" % result["loss"][-50:].mean())
print("Total time: %.1f s   |   per frame: %.4f s"
      % (s["total_time_s"], s["time_per_frame_s"]))
print("Saved numbers to:", folder)
print("Saved graph to:  ", figure)
