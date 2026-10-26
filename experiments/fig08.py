"""
fig08.py - reproduces Fig. 8 of the paper (Section 5.1, WDs switching off/on).

Paper setup: N = 10 WDs, K = 10, 10,000 time frames. After DROO converges:
    frames 6,000 / 6,500 / 7,000 / 7,500 : one WD switched OFF each time
    frames 8,000 / 8,500 / 9,000          : WDs switched back ON
    frame  9,500                          : two WDs switched OFF
The schedule and the switching logic are copied from the authors'
DROO/demo_on_off.py:
  - a switched-off WD's channel gain is divided by 1e6 (practically 0) in
    the network's input, so the network keeps its 10 inputs
  - only the active WDs are passed to the solver (P2)
  - the WD with the highest index is switched first (the paper says
    "randomly"; the code always picks the last active WD)
  - at frame 9,000 the code switches TWO WDs back on, so all 10 are active
    again (the paper's text lists only three "on" moments)
  - the reference rate is re-loaded from data_<number of active WDs>.mat

As for Fig. 6: the paper's caption says K = 10, the authors' demo uses
adaptive K (Delta = 32). We follow the paper by default; add --adaptive
for the demo's setting.

Paper's claims to check:
  - the training loss increases only a little after WDs switch off/on
  - the moving average of the normalized rate stays above 0.99

How to run (from the project folder):
    python experiments/fig08.py               # paper setting (fixed K = 10)
    python experiments/fig08.py --adaptive    # authors' demo setting
    python experiments/fig08.py --frames 500  # quick test

Outputs:
    results/fig08/<tag>.npz and <tag>_summary.json
    figures/fig08_<tag>.png
"""

import argparse

from common import (FIGURES_DIR, load_data, plot_loss_and_rate, rolling,
                    run_droo, save_result)

parser = argparse.ArgumentParser(description="Reproduce Fig. 8")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--frames", type=int, default=10000, help="time frames")
parser.add_argument("--adaptive", action="store_true",
                    help="use adaptive K (Delta=32) like the authors' demo")
args = parser.parse_args()

N = 10
n = args.frames
delta = 32 if args.adaptive else None

# (fraction of the run, action, how many WDs) -- same as demo_on_off.py
SCHEDULE = [(0.60, "off", 1), (0.65, "off", 1), (0.70, "off", 1),
            (0.75, "off", 1), (0.80, "on", 1), (0.85, "on", 1),
            (0.90, "on", 2), (0.95, "off", 2)]
EVENTS = {int(f * n): (action, count) for f, action, count in SCHEDULE}


def wd_off(state):
    """Same as the authors' WD_off(): switch off the last active WD."""
    if state["N_active"] > 5:                  # authors allow at most 5 off
        state["N_active"] -= 1
        k = state["N_active"]
        state["channel"][:, k] = state["channel"][:, k] / 1000000
        print("    WD %d switched OFF" % (k + 1))


def wd_on(state):
    """Same as the authors' WD_on(): switch the last switched-off WD back on."""
    if state["N_active"] < N:
        state["N_active"] += 1
        k = state["N_active"] - 1
        state["channel"][:, k] = state["channel"][:, k] * 1000000
        print("    WD %d switched ON" % (k + 1))


def switch_wds(i, state):
    """Called at the start of every frame; applies the on/off schedule."""
    if i in EVENTS:
        action, count = EVENTS[i]
        print("  frame %d:" % i)
        for _ in range(count):
            wd_off(state) if action == "off" else wd_on(state)
        # reference ("best known") rate for this number of active WDs
        _, state["rate"] = load_data("data_%d" % state["N_active"])
        print("    now %d active WDs" % state["N_active"])


print("Fig. 8: N=%d, K=%s, %d frames, seed %d"
      % (N, "adaptive (Delta=32)" if args.adaptive else "10 (fixed)", n, args.seed))

result = run_droo(N=N, n=n, K=N, delta=delta, seed=args.seed,
                  hook=switch_wds, split=False)

first = min(EVENTS)
after = result["rate_ratio"][first:]
mean50, _, _ = rolling(result["rate_ratio"], 50)
result["summary"]["min_rate_after_first_event"] = float(after.min())
result["summary"]["min_moving_avg_after_first_event"] = float(mean50[first:].min())
result["summary"]["event_frames"] = sorted(EVENTS)

tag = "seed%d" % args.seed
if args.adaptive:
    tag += "_adaptive"
if n != 10000:
    tag += "_frames%d" % n

folder = save_result("fig08", tag, result)
figure = plot_loss_and_rate(
    result["loss"], result["rate_ratio"], train_interval=10,
    path=FIGURES_DIR / ("fig08_%s.png" % tag),
    title="Fig. 8 replication: WDs off/on (N=10, K=%s, seed %d)"
          % ("adaptive" if args.adaptive else "10", args.seed),
    events=[(f, ("%s x%d" % (a, c)) if c > 1 else a)
            for f, (a, c) in sorted(EVENTS.items())])

s = result["summary"]
print()
print("Average normalized rate (last %d frames): %.5f"
      % (result["config"]["num_test"], s["avg_normalized_rate_test"]))
print("After frame %d: lowest single rate %.4f"
      % (first, s["min_rate_after_first_event"]))
print("After frame %d: lowest moving average %.4f (paper: > 0.99)"
      % (first, s["min_moving_avg_after_first_event"]))
print("Total time: %.1f s   |   per frame: %.4f s"
      % (s["total_time_s"], s["time_per_frame_s"]))
print("Saved numbers to:", folder)
print("Saved graph to:  ", figure)
