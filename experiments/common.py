"""
common.py - shared code for every experiment in this replication.

Every experiment script (fig05.py, fig06.py, ...) imports from here, so the
DROO loop, the data loading, the saving and the plotting are written ONCE.

What this file does NOT do: change the authors' algorithm. It imports their
MemoryDNN (DROO/memoryPyTorch.py) and their solver (DROO/optimization.py)
unchanged, and repeats the main loop of DROO/mainPyTorch.py line for line.
The only additions are: fixed random seeds, saving results to files, and
saving graphs to files instead of opening pop-up windows.
"""

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

import matplotlib
matplotlib.use("Agg")  # draw graphs straight into files, no pop-up windows
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Folders (worked out from where this file lives, so it runs from anywhere)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DROO_DIR = PROJECT_ROOT / "DROO"
DATA_DIR = DROO_DIR / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# Quick-test mode: set the environment variable DROO_QUICK=1 (run_all.py
# --quick does this) to run every experiment with tiny frame counts, writing
# to quicktest/ instead of results/ and figures/. Only for checking that
# everything runs; the numbers mean nothing.
QUICK = bool(os.environ.get("DROO_QUICK"))
if QUICK:
    RESULTS_DIR = PROJECT_ROOT / "quicktest" / "results"
    FIGURES_DIR = PROJECT_ROOT / "quicktest" / "figures"


def frames(n, quick_n=400):
    """Number of frames to use: n normally, quick_n in quick-test mode."""
    return quick_n if QUICK else n

# Let Python find the authors' files (memoryPyTorch.py, optimization.py).
sys.path.insert(0, str(DROO_DIR))

import torch                                  # noqa: E402
from memoryPyTorch import MemoryDNN           # noqa: E402  (authors' code)
from optimization import bisection            # noqa: E402  (authors' code)

# One CPU thread: the network is tiny, so extra threads only add overhead
# (measured: ~8% faster on a Ryzen 9). This does not change any result.
torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    """Fix all randomness so the same seed always gives the same run."""
    np.random.seed(seed)     # used by the replay-memory sampling
    torch.manual_seed(seed)  # used by the network's starting weights


def load_data(name):
    """
    Load one of the authors' data files, e.g. name='data_10'.
    Returns:
        channel: channel gains, already multiplied by 1e6 (as in mainPyTorch.py)
        rate:    the reference ("best known") computation rate for each sample
    """
    mat = sio.loadmat(str(DATA_DIR / name))
    channel = mat["input_h"] * 1000000
    rate = mat["output_obj"]
    return channel, rate


# ---------------------------------------------------------------------------
# The DROO loop (same logic as DROO/mainPyTorch.py)
# ---------------------------------------------------------------------------
def run_droo(N=10, n=10000, K=None, delta=None, decoder="OP",
             memory=1024, batch_size=128, train_interval=10, lr=0.01,
             seed=0, data_name=None, hook=None, split=True, verbose=True):
    """
    Run DROO once and return everything it recorded.

    N              number of wireless devices (WDs)
    n              number of time frames
    K              number of candidate actions per frame (starts at N if None)
    delta          update interval for adaptive K (paper Sec. 4.4);
                   None = keep K fixed for the whole run
    decoder        'OP' (order-preserving, the paper's method) or 'KNN'
    memory         replay memory size
    batch_size     training batch size
    train_interval train the network every this many frames (delta in paper)
    lr             learning rate of the Adam optimizer
    seed           random seed
    data_name      data file to use (default: 'data_<N>')
    hook           optional function hook(i, state), called at the start of
                   every frame; demos use it to change weights or switch WDs
                   off/on by editing state['channel'], state['rate'] and
                   state['weights'].
    split          True  = 80/20 train/test split, as in mainPyTorch.py
                   False = frame i uses data sample i, as in the two demos
                           (demo_alternate_weights.py, demo_on_off.py)
    """
    set_seed(seed)
    if K is None:
        K = N
    K_initial = K
    if data_name is None:
        data_name = "data_%d" % N
    # The authors' decode() compares the mode with "is", so pass the exact
    # same string object Python uses for the literal 'OP' / 'KNN'.
    decoder = sys.intern(decoder)

    channel, rate = load_data(data_name)

    # Same 80/20 train/test split as mainPyTorch.py
    split_idx = int(.8 * len(channel))
    num_test = min(len(channel) - split_idx, n - int(.8 * n))

    mem = MemoryDNN(net=[N, 120, 80, N],
                    learning_rate=lr,
                    training_interval=train_interval,
                    batch_size=batch_size,
                    memory_size=memory)

    # Things a hook may change during the run:
    #   channel, rate  -> data in use (rate = "best known" reference rate)
    #   weights        -> WD weights ([] = authors' default [1, 1.5, 1, ...])
    #   N_active       -> only the first N_active WDs go into the solver
    #                     (the others are "switched off", as in demo_on_off.py)
    state = {"channel": channel, "rate": rate, "weights": [], "N_active": N}

    rate_his, rate_ratio, k_idx_his, K_his, mode_his = [], [], [], [], []

    start = time.perf_counter()
    cpu_start = time.process_time()
    for i in range(n):
        if hook is not None:
            hook(i, state)

        if verbose and i % (n // 10) == 0:
            print("  progress: %3d%%" % (100 * i // n), flush=True)

        # Adaptive K (paper Sec. 4.4), written exactly as in mainPyTorch.py
        if delta is not None and i > 0 and i % delta == 0:
            if delta > 1:
                max_k = max(k_idx_his[-delta:-1]) + 1
            else:
                max_k = k_idx_his[-1] + 1
            K = min(max_k + 1, N)

        if not split:
            i_idx = i                                 # demos: sample i
        elif i < n - num_test:
            i_idx = i % split_idx                     # training frames
        else:
            i_idx = i - n + num_test + split_idx      # test frames

        h = state["channel"][i_idx, :]

        # 1) the network proposes K candidate actions
        m_list = mem.decode(h, K, decoder)

        # 2) solve (P2) for each candidate, keep the best
        #    (with all WDs active this is exactly mainPyTorch.py's call)
        Na = state["N_active"]
        w = state["weights"][:Na] if len(state["weights"]) else []
        r_list = [bisection(h[:Na] / 1000000, m[:Na], w)[0] for m in m_list]
        best = int(np.argmax(r_list))

        # 3) store the best (h, action) pair and train every train_interval
        mem.encode(h, m_list[best])

        rate_his.append(r_list[best])
        rate_ratio.append(r_list[best] / state["rate"][i_idx][0])
        k_idx_his.append(best)
        K_his.append(K)
        mode_his.append(m_list[best])

    total_time = time.perf_counter() - start
    cpu_time = time.process_time() - cpu_start

    rate_ratio = np.asarray(rate_ratio)
    return {
        "config": dict(N=N, n=n, K_initial=K_initial, delta=delta, decoder=decoder, memory=memory,
                       batch_size=batch_size, train_interval=train_interval,
                       lr=lr, seed=seed, data_name=data_name,
                       split=split, num_test=num_test),
        "rate": np.asarray(rate_his),
        "rate_ratio": rate_ratio,
        "k_idx": np.asarray(k_idx_his),
        "K": np.asarray(K_his),
        "mode": np.asarray(mode_his),
        "loss": np.asarray(mem.cost_his),
        "summary": {
            # mean over the test frames (mainPyTorch.py prints almost the
            # same number; it skips the very last frame by accident)
            "avg_normalized_rate_test": float(rate_ratio[-num_test:].mean()),
            "total_time_s": total_time,
            "time_per_frame_s": total_time / n,
            # CPU time used by this run only (fairer when several runs
            # share the machine; Table 2 reports "CPU execution latency")
            "cpu_time_s": cpu_time,
            "cpu_time_per_frame_s": cpu_time / n,
        },
    }


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------
def environment_info():
    """Record what the run happened on, so the report can say it."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cpu": platform.processor(),
        "os": platform.platform(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save_result(experiment, tag, result):
    """
    Save one run to results/<experiment>/:
        <tag>.npz           all recorded numbers (for re-plotting later)
        <tag>_summary.json  settings + headline numbers (human readable)
    """
    folder = RESULTS_DIR / experiment
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / (tag + ".npz"),
                        rate=result["rate"], rate_ratio=result["rate_ratio"],
                        k_idx=result["k_idx"], K=result["K"],
                        mode=result["mode"], loss=result["loss"])
    with open(folder / (tag + "_summary.json"), "w") as f:
        json.dump({"config": result["config"],
                   "summary": result["summary"],
                   "environment": environment_info()}, f, indent=2)
    return folder


def load_result(experiment, tag):
    """Load a saved run back (used to re-draw graphs without re-running)."""
    folder = RESULTS_DIR / experiment
    data = dict(np.load(folder / (tag + ".npz")))
    with open(folder / (tag + "_summary.json")) as f:
        data.update(json.load(f))
    return data


# ---------------------------------------------------------------------------
# Plotting (paper style: rolling mean + min/max band over 50 frames)
# ---------------------------------------------------------------------------
def rolling(x, window=50):
    """Rolling mean, min and max over the last `window` frames."""
    x = np.asarray(x, dtype=float)
    mean = np.empty_like(x)
    lo = np.empty_like(x)
    hi = np.empty_like(x)
    for t in range(len(x)):
        w = x[max(0, t - window + 1): t + 1]
        mean[t], lo[t], hi[t] = w.mean(), w.min(), w.max()
    return mean, lo, hi


def moving_average_plot(curves, path, title, ylabel="Moving average of normalized rate",
                        window=200, ylim=None, xlabel="Time frame t"):
    """
    Several moving-average curves on one graph (style of Figs. 9-12).
    curves: list of (label, values)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, values in curves:
        mean, _, _ = rolling(values, window)
        ax.plot(np.arange(1, len(mean) + 1), mean, linewidth=1.1, label=label)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loss_and_rate(loss, rate_ratio, train_interval, path,
                       title=None, events=None, window=50, rate_ylim=(0.8, 1.0)):
    """
    Two stacked graphs like the paper's Fig. 5/6/8:
      top:    training loss
      bottom: normalized computation rate (rolling mean + min/max band)
    events: optional list of (frame, label) to mark with arrows/lines.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    t_loss = np.arange(1, len(loss) + 1) * train_interval
    ax1.plot(t_loss, loss, color="tab:red", linewidth=0.8)
    ax1.set_ylabel("Training loss")
    ax1.grid(alpha=0.3)

    t = np.arange(1, len(rate_ratio) + 1)
    mean, lo, hi = rolling(rate_ratio, window)
    ax2.fill_between(t, lo, hi, color="tab:blue", alpha=0.2, linewidth=0)
    ax2.plot(t, mean, color="tab:blue", linewidth=1.2)
    ax2.set_ylim(*rate_ylim)
    ax2.set_ylabel("Normalized computation rate")
    ax2.set_xlabel("Time frame t")
    ax2.grid(alpha=0.3)

    for frame, label in (events or []):
        for ax in (ax1, ax2):
            ax.axvline(frame, color="gray", linestyle="--", linewidth=0.8)
        ax1.annotate(label, (frame, ax1.get_ylim()[1]), fontsize=8,
                     ha="center", va="bottom", annotation_clip=False)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
