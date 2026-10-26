"""
fig09.py - reproduces Fig. 9 of the paper (Section 5.1, algorithm parameters).

Moving average of the normalized rate for N = 10 when ONE parameter is
changed at a time (all others at the paper's defaults: memory 1024,
batch 128, training interval 10, learning rate 0.01, K = N = 10):
  (a) memory size       128, 256, 512, 1024, 2048
  (b) batch size        32, 64, 128, 512, 1024
  (c) training interval 5, 10, 20, 50, 100
  (d) learning rate     0.1, 0.01, 0.001, 0.0001, 0.00001
The default run (1024 / 128 / 10 / 0.01) is shared by all four panels and is
the same configuration as fig05.py, so it should give the same numbers.

Paper's claims to check:
  (a) memory 128 fluctuates more; 2048 needs more data to converge
  (b) batch 32 is slower to converge; 1024 also worse ("old" data reused)
  (c) shorter interval converges faster; 50 and 100 clearly slower
  (d) learning rates 0.1 and 0.0001/0.00001 converge to worse solutions

How to run (from the project folder):
    python experiments/fig09.py --run memory=128    # one run
    python experiments/fig09.py --plot               # draw the 4 panels
    python experiments/run_all.py                    # everything, in parallel
"""

import argparse

from common import (frames, FIGURES_DIR, RESULTS_DIR, load_result, moving_average_plot,
                    run_droo, save_result)

EXPERIMENT = "fig09"
DEFAULTS = {"memory": 1024, "batch": 128, "interval": 10, "lr": 0.01}
PANELS = {
    "memory": [128, 256, 512, 1024, 2048],
    "batch": [32, 64, 128, 512, 1024],
    "interval": [5, 10, 20, 50, 100],
    "lr": [0.1, 0.01, 0.001, 0.0001, 0.00001],
}
LABELS = {"memory": "Memory size", "batch": "Batch size",
          "interval": "Training interval", "lr": "Learning rate"}
N, SEED = 10, 0
FRAMES = frames(10000)


def tag_of(name, value):
    if value == DEFAULTS[name]:
        return "default"
    return "%s_%s" % (name, value)


# Every distinct run (the default is listed once)
JOBS = ["default"] + ["%s=%s" % (k, v) for k, vals in PANELS.items()
                      for v in vals if v != DEFAULTS[k]]


def job_tag(job):
    """'memory=128' -> 'memory_128' (the file name of its results)."""
    return job.replace("=", "_")


def run(job):
    settings = dict(DEFAULTS)
    if job != "default":
        name, value = job.split("=")
        settings[name] = float(value) if name == "lr" else int(value)
    print("Fig. 9 run:", settings)
    result = run_droo(N=N, n=FRAMES, K=N, delta=None, seed=SEED,
                      memory=settings["memory"], batch_size=settings["batch"],
                      train_interval=settings["interval"], lr=settings["lr"])
    save_result(EXPERIMENT, job_tag(job), result)
    s = result["summary"]
    print("Average normalized rate (test frames): %.5f   time %.0f s"
          % (s["avg_normalized_rate_test"], s["total_time_s"]))


def plot():
    for panel, letter in zip(PANELS, "abcd"):
        curves = []
        for value in PANELS[panel]:
            r = load_result(EXPERIMENT, tag_of(panel, value))
            curves.append(("%s = %s" % (LABELS[panel], value), r["rate_ratio"]))
        path = moving_average_plot(
            curves, FIGURES_DIR / ("fig09%s_%s.png" % (letter, panel)),
            title="Fig. 9(%s) replication: %s (N=10)" % (letter, LABELS[panel].lower()),
            ylim=(0.92, 1.0))
        print("Saved", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Fig. 9")
    parser.add_argument("--run", help="one job, e.g. memory=128 or default")
    parser.add_argument("--plot", action="store_true", help="draw the figures")
    args = parser.parse_args()
    if args.run:
        run(args.run)
    if args.plot:
        plot()
