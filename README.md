# DROO replication

A replication of

> L. Huang, S. Bi, and Y. J. Zhang, "Deep Reinforcement Learning for Online
> Computation Offloading in Wireless Powered Mobile-Edge Computing Networks,"
> *IEEE Transactions on Mobile Computing*, vol. 19, no. 11, pp. 2581-2593,
> Nov. 2020. (arXiv:1808.01977, in `paper/`)

using the authors' code from <https://github.com/revenol/DROO>, at commit
`563d6c7723463c3e6b77b470e3200268c5fad4f3` (the PyTorch version).

Every figure and table with results in the paper is reproduced: **Fig. 4-15
and Table 2**. The comparison between the paper's results and ours is in
**[REPORT.md](REPORT.md)**.

---

## Folder layout

```
DROO-replication/
├── README.md            this file: setup, how to run, every change we made
├── REPORT.md            paper vs. our results, figure by figure
├── requirements.txt     exact library versions (pip freeze)
├── paper/               the paper (PDF)
├── DROO/                the authors' code and data (only the 2 fixes below)
├── baselines/           code we wrote: benchmark methods + a fast solver
├── experiments/         one script per paper result + the shared code
├── results/             saved numbers of every run (.npz + _summary.json)
└── figures/             every replicated figure (.png)
```

## Setup (Windows, PowerShell)

```
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Tested with Python 3.12, PyTorch 2.14 (CPU), NumPy 2.5, SciPy 1.18,
CVXPY 1.9. Everything runs on the CPU, like the paper (Intel i5-4570).

## How to run

All commands are run from the project folder.

| Paper result | Command | Time (Ryzen 9, alone) |
|---|---|---|
| Fig. 5 | `python experiments/fig05.py` | ~5 min |
| Fig. 6 | `python experiments/fig06.py` (paper, K=10) and `--adaptive` (authors' demo) | ~5 + 3 min |
| Fig. 8 | `python experiments/fig08.py` and `--adaptive` | ~5 + 3 min |
| Fig. 4, 7, 9, 10, 11, 12, 13, 14, 15, Table 2 | `python experiments/run_all.py` (54 runs, 11 at a time) | ~35 min |
| Redraw all figures from saved results | `python experiments/run_all.py --plot-only` | ~1 min |
| Check that everything works (tiny runs) | `python experiments/run_all.py --quick` | ~3 min |
| Check our fast solver against the authors' | `python baselines/test_p2_vectorized.py` | ~1 min |
| Bonus: CPU vs GPU (in `.venv-gpu`, see below) | `python experiments/gpu_benchmark.py` | ~15 min |
| **Averaged figures** (every figure over 10 seeds, see below) | `python experiments/averaged.py` | ~1 h on 4 cores |

Each run saves `results/<experiment>/<run>.npz` (all recorded numbers) and
`<run>_summary.json` (settings, headline numbers, machine info). Graphs are
drawn from these files, so a graph can be redrawn without re-running.
`run_all.py` skips runs whose results already exist, so it can be stopped
and restarted.

## Averaged figures (10 seeds)

`experiments/averaged.py` re-runs every figure's setting with seeds 0-9 and
draws one clean figure per paper figure in `figures/averaged/`: each setting
is the **mean over the 10 seeds** of its moving average, with a light band of
**+/-1 standard error** (Fig. 13/14: error bars = +/-1 std). Fig. 9's four
panels are combined into one 2x2 figure, and the sweeps (Figs. 9-12) use a
500-frame moving average to make the lines smooth. The per-setting mean, std,
worst and best seed are in `results/averaged/summary.md`.

To make 390 runs affordable these runs use `solver="vectorized"` in
`run_droo()` (the authors' (P2) equations solved for all K candidates at
once, `baselines/p2_vectorized.py`). It picks exactly the same actions as
the authors' `bisection()`, about 8x faster. The Fig. 11-13 runs keep
`bisection()`, because Fig. 13 plots CPU time. The seed-independent
baselines of Fig. 14/15 (CD, LR, Edge, Local) and the Table 2 timings are
reused from the single-seed runs.

Options: `--seeds 5`, `--only fig09 fig10`, `--plot-only`, `--dry-run`.

## Bonus GPU environment

The replication itself uses the CPU-only environment above. The GPU test
uses a second environment, so the CPU setup stays unchanged:

```
python -m venv .venv-gpu
.venv-gpu\Scripts\Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install numpy scipy matplotlib pandas
python experiments/gpu_benchmark.py
```
(Tested with NVIDIA driver 591.84 / CUDA 13.1, RTX 5070 Ti Laptop GPU.)

## Randomness

Every run fixes the random seed (seed 0) for NumPy and PyTorch, so the same
script gives the same result every time on the same machine. Windows and
Linux can differ in the last digits (different maths libraries), e.g. the
500-frame Fig. 5 test gives 0.97695 on Windows and 0.97764 on Linux.

---

## Changes to the authors' code (`DROO/`)

Kept to the minimum needed to run on current libraries. Each is its own git
commit, so `git log -p DROO/` shows the exact lines.

| # | File(s) | Change | Why |
|---|---|---|---|
| 1 | 4 files | `mpl.style.use('seaborn')` -> `'seaborn-v0_8'` | the style was renamed in matplotlib 3.6; the old name crashes |
| 2 | `demo_alternate_weights.py`, `demo_on_off.py` | `from memory import ...` -> `from memoryPyTorch import ...` | as instructed in the authors' README for PyTorch users |
| - | `requirements.txt` | added `pandas` | used by `plot_rate()` but missing from the authors' README |

The algorithm itself (`memoryPyTorch.py`, `optimization.py`) is unchanged.

## What we added

**`experiments/common.py`** - shared code for all experiments. `run_droo()`
repeats the main loop of `DROO/mainPyTorch.py` line for line, using the
authors' `MemoryDNN` and `bisection()` unchanged. It only adds: fixed seeds,
saving results, saving graphs to files (no pop-up windows), recording CPU
time, a `hook` so experiments can change weights or switch devices mid-run,
and `split=False` for the demos' data order (frame i uses sample i).
It also sets PyTorch to one thread (the network is tiny; more threads only
add overhead - measured ~8% faster; results unchanged).

**`baselines/methods.py`** - the paper's benchmark methods:
- **CD**: the authors' own `cd_method()` from `DROO/optimization.py`
- **Edge Computing** / **Local Computing**: all-ones / all-zeros action,
  solved with the authors' `bisection()`
- **LR**: *not in the authors' repository*, written from the paper's
  description (Sec. 5.3) - see "Interpretations" below

**`baselines/p2_vectorized.py`** - the authors' (P2) solver rewritten to
solve many actions at once. Same equations, constants and stopping rule;
`test_p2_vectorized.py` checks it matches `bisection()` to ~1e-16 and
reproduces the stored optimum. Used where all 1,024 actions must be tried
(the true optimum for Fig. 7) and for per-device rates (Fig. 7).

## Interpretations and deviations (also discussed in REPORT.md)

1. **LR baseline.** Taken literally, (P1) with x relaxed to [0,1] is not
   convex. We use the standard convex form that equals (P1) for binary x:
   local term `(1-x_i)^(2/3) a^(1/3)` (geometric mean), offloading term
   `tau_i log(1 + eta2 h_i^2 a / tau_i)` with `tau_i <= x_i`; solved with
   CVXPY (Clarabel), then rule (12), then the binary action is evaluated
   with the authors' solver.
2. **Fixed vs. adaptive K in the demos.** The captions of Fig. 6 and 8 say
   K = 10, while the authors' demos use adaptive K (Delta = 32). We ran
   both; the paper's setting is the default.
3. **Fig. 8 details from the authors' code:** the device switched off is
   always the one with the highest index (the paper says "randomly"), and
   at frame 9,000 two devices are switched back on (all 10 active again).
4. **Reference rates.** For N = 10 the stored `output_obj` is the true
   optimum (exhaustive search); for N = 20, 30 it is CD's result (authors'
   data README). So "CD" in Fig. 14 is the stored value, as in the paper,
   and CD's run time is measured on the first 100 test channels.
5. **Table 2, fixed K:** timed over 5,000 frames instead of 30,000 (the
   cost per frame does not shrink with a fixed K). Timing is CPU time
   (`time.process_time`); runs made by `run_all.py` share the machine, which
   inflates the times - see REPORT.md.
6. **PyTorch version vs. the paper's TensorFlow 1 code:** `memoryPyTorch.py`
   creates a new Adam optimizer at every training step (TF1 creates it once)
   and uses beta1 = 0.09. This may explain why our training loss settles
   around 0.08-0.10 instead of the paper's 0.04. Kept as is.

## Git history

```
git log --oneline
```
shows the steps: original code -> fixes -> one commit per replicated figure.

## AI assistance

The experiment scripts, baselines and these documents were written with the
help of Claude (Anthropic); commits made together are marked with a
`Co-Authored-By` line. All runs were made on the author's own computer.
