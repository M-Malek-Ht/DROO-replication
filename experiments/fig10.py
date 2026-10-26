"""
fig10.py - reproduces Fig. 10 of the paper (Section 5.1, quantization method).

Moving average (window 200) of the normalized rate for N = 10, comparing the
paper's order-preserving quantization ('OP') with KNN quantization, for
K = 1, 2 and 10 (fixed K, no adaptive K). Both quantizers are the authors'
code (MemoryDNN.knm and MemoryDNN.knn in DROO/memoryPyTorch.py).

Paper's claims to check:
  - K = 10: both methods approach 1
  - K = 2: both converge to only about 0.95; OP converges faster than KNN
  - K = 1: neither converges (no exploration without action choices)

How to run (from the project folder):
    python experiments/fig10.py --run OP_K2
    python experiments/fig10.py --plot
"""

import argparse

from common import (frames, FIGURES_DIR, load_result, moving_average_plot, run_droo,
                    save_result)

EXPERIMENT = "fig10"
N, SEED = 10, 0
FRAMES = frames(10000)
JOBS = ["%s_K%d" % (d, k) for k in (10, 2, 1) for d in ("OP", "KNN")]


def job_tag(job):
    return job


def run(job):
    decoder, k = job.split("_K")
    print("Fig. 10 run: decoder %s, K = %s" % (decoder, k))
    result = run_droo(N=N, n=FRAMES, K=int(k), delta=None, decoder=decoder,
                      seed=SEED)
    save_result(EXPERIMENT, job, result)
    s = result["summary"]
    print("Average normalized rate (test frames): %.5f   time %.0f s"
          % (s["avg_normalized_rate_test"], s["total_time_s"]))


def plot():
    names = {"OP": "Order-preserving", "KNN": "KNN"}
    curves = []
    for job in JOBS:
        decoder, k = job.split("_K")
        # KNN dashed: for K = 1 and K = 2 the two methods choose exactly the
        # same candidates, so their curves lie on top of each other
        curves.append(("%s, K = %s" % (names[decoder], k),
                       load_result(EXPERIMENT, job)["rate_ratio"],
                       "--" if decoder == "KNN" else "-"))
    path = moving_average_plot(curves, FIGURES_DIR / "fig10_quantization.png",
                               title="Fig. 10 replication: quantization methods (N=10)",
                               ylim=(0.65, 1.0))
    print("Saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Fig. 10")
    parser.add_argument("--run", help="one job, e.g. OP_K2 or KNN_K10")
    parser.add_argument("--plot", action="store_true", help="draw the figure")
    args = parser.parse_args()
    if args.run:
        run(args.run)
    if args.plot:
        plot()
