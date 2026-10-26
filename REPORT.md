# Replication report: DROO (Huang, Bi & Zhang, IEEE TMC 2020)

**Paper:** "Deep Reinforcement Learning for Online Computation Offloading in
Wireless Powered Mobile-Edge Computing Networks" (arXiv:1808.01977).
**Code:** authors' PyTorch version, github.com/revenol/DROO @ `563d6c7`,
with two compatibility fixes only (see README.md).
**Machine:** AMD Ryzen 9 (Windows 11), Python 3.12, PyTorch 2.14 (CPU only).
**Seed:** 0 for every run. All numbers below come from `results/`, all
graphs are in `figures/`.

## Summary

Every result in the paper was re-run: **Fig. 4-15 and Table 2**, plus a
bonus CPU-vs-GPU test.

| Paper result | What it claims | Replicated? |
|---|---|---|
| Fig. 4 | best action is among the first 5 candidates after t = 5,000 | ✅ 99.8% of frames |
| Fig. 5 | DROO converges to ~optimal (Q̂ → 1); loss ~0.04 | ✅ rate; ⚠️ loss ~0.10 |
| Fig. 6 | adapts to swapped weights; moving avg > 0.99 | ✅ |
| Fig. 7 | adapts to temporarily raised weights | ✅ (WD1's gain larger than reported) |
| Fig. 8 | copes with devices switching off/on; moving avg > 0.99 | ✅ with K = 10; ❌ with the demo's adaptive K |
| Fig. 9 | effect of memory, batch, training interval, learning rate | ✅ (c), (d); ~ (a), (b) |
| Fig. 10 | order-preserving quantization beats KNN | ~ only partly; identical for K ≤ 2 |
| Fig. 11-13 | adaptive K: Δ ≥ 16 needed; Δ = 32 good trade-off | ✅ |
| Fig. 14 | DROO ≈ CD > LR > Edge > Local | ✅ numbers within ~1-2% of the bars |
| Fig. 15 | DROO's median ≈ 1, better than LR | ✅ |
| Table 2 | DROO is 1-2 orders of magnitude faster than CD | ✅ (CD 73× slower at N = 30; paper 65×) |
| Table 2 | DROO faster than LR | ❌ not with a modern LR solver - see Table 2 |

The paper's central claim holds: **DROO reaches near-optimal computation
rates while being far faster than coordinate descent**, and it adapts
online to changes in weights and active devices.

---

## Fig. 4 - rank of the best action (`figures/fig04_kstar.png`)

Same run as Fig. 5 (N = 10, K = 10). Paper: "the selected indices k*_t are no
larger than 5 when t ≥ 5,000". **Ours:** after t = 5,000, k* ≤ 5 in 99.82% of
frames (largest value 7). The 50-frame mean settles around 1.5, as in the
paper's figure. ✅ (near-exact; "no larger than 5" holds for all but a
handful of frames).

## Fig. 5 - convergence (`figures/fig05_seed0.png`)

N = 10, K = 10, 10,000 frames.

| | Paper | Ours |
|---|---|---|
| 50-frame moving average ≥ 0.98 | from t > 400 | from t = 171; 99.3% of frames after t = 400 |
| Average Q̂ on test frames | → 1 | **0.99955** |
| Training loss at the end | ~0.04 | ~0.10 (lowest 0.045) |

✅ for the rate. ⚠️ for the loss: ours settles ~2.5× higher. Likely cause:
the PyTorch file re-creates the Adam optimizer at every training step (the
TF1 original creates it once) and uses beta1 = 0.09. Supporting evidence:
in Fig. 9(d) the run with learning rate 0.001 ends with loss **0.042**, the
paper's value. The decisions are unaffected (rate 0.9996).

Running the original `mainPyTorch.py` (adaptive K, 30,000 frames) also
gave 0.99950 (Windows) and 0.99945 (Colab), so our script reproduces the
authors' own program.

## Fig. 6 - alternating weights (`figures/fig06_seed0*.png`)

All weights swapped at t = 6,000 and back at 8,000.

| | Paper (K = 10) | Ours, K = 10 | Ours, adaptive K (authors' demo) |
|---|---|---|---|
| Lowest 50-frame moving avg after t = 6,000 | > 0.99 | **0.9970** ✅ | **0.9932** ✅ |
| Lowest single-frame Q̂ after t = 6,000 | > 0.95 | 0.9492 (just below) | 0.9019 |
| Average Q̂, last 2,000 frames | - | 0.99911 | 0.99916 |
| Time per frame | - | 0.030 s | 0.015 s |

✅ The loss rises after each swap and settles again, as in the paper.
The single-frame minimum misses 0.95 by 0.001 with K = 10. With adaptive
K it dips lower: right after a change, few candidates are tried, so an
unlucky frame has fewer chances to find a good action.

## Fig. 7 - temporarily raised weights (`figures/fig07_new_weights.png`)

*Not in the authors' repository - written from the paper's description.*
WD2's weight 1.5 → 3 at t = 4,000; WD1's weight 1 → 3 at t = 6,000; both
reset at 8,000. The true optimum under the changed weights was found by
trying all 1,024 actions (`baselines/p2_vectorized.py`, verified against the
authors' solver to ~1e-16).

| | Paper | Ours |
|---|---|---|
| Relative rates in the first 3,000 frames | mostly below 1 | WD1 0.91, WD2 0.98 (mean) ✅ |
| WD2 gain while doubled (4,000-8,000) | +12% | **+13.2%** ✅ |
| WD1 gain while tripled (6,000-8,000) | +26% | **+36.8%** ⚠️ |
| WD2's peak relative rate | "2 to 3 times" | 3.3× (single frame) ✅ |
| Back to ~1 after reset | yes | yes ✅ |
| Lowest 50-frame moving avg of Q̂ after 4,000 | "quickly adapts" | 0.9959 ✅ |

The qualitative behaviour matches. WD1's gain is larger than reported; the
paper does not define exactly how the averages were taken (e.g. per-frame
ratio vs. ratio of sums), which may explain the difference.

## Fig. 8 - devices switching off and on (`figures/fig08_seed0*.png`)

Switch-offs at 6,000 / 6,500 / 7,000 / 7,500; back on at 8,000 / 8,500 /
9,000 (two at 9,000, per the authors' code); two off at 9,500.

| | Paper (K = 10) | Ours, K = 10 | Ours, adaptive K (authors' demo) |
|---|---|---|---|
| Lowest 50-frame moving avg after 6,000 | > 0.99 | **0.9903** ✅ | 0.9788 ❌ |
| Training loss after events | increases little | small bumps ✅ | small bumps ✅ |
| Average Q̂, last 2,000 frames | - | 0.99789 | 0.99443 |

✅ with the paper's setting. **New observation:** switching devices **off**
barely matters; the dips come when devices come back **on** (8,500, 9,000)
and at the double switch-off (9,500). With adaptive K, DROO reacts more
slowly to such changes - a trade-off the paper does not discuss.
Code details differing from the text: the device switched off is always the
highest-numbered active one (paper: "randomly").

## Fig. 9 - algorithm parameters (`figures/fig09*.png`)

N = 10, K = 10, one parameter changed at a time. "MA" = 200-frame moving
average of Q̂; "≥ 0.99 at" = first frame where MA reaches 0.99.

| Run | Avg Q̂ (test) | MA at t = 2,000 | ≥ 0.99 at | Paper's claim | |
|---|---|---|---|---|---|
| default (1024 / 128 / 10 / 0.01) | 0.99955 | 0.9931 | 1,119 | - | identical to Fig. 5 run ✅ |
| memory 128 | 0.99935 | 0.9966 | 839 | larger fluctuations | ❌ not seen |
| memory 2048 | 0.99960 | 0.9918 | 1,122 | needs more data | ~ marginally slower |
| batch 32 | 0.99702 | 0.9908 | 1,855 | worse | ✅ slower, lowest final |
| batch 1024 | 0.99967 | 0.9899 | 1,950 | worse (stale data) | ~ slower start, fine at end |
| interval 5 | 0.99972 | 0.9931 | 1,550 | faster | ✅ best by t = 4,000 |
| interval 50 | 0.99469 | 0.9828 | 3,947 | slower | ✅ |
| interval 100 | 0.99176 | 0.9834 | 5,659 | much slower | ✅ |
| lr 0.1 | 0.92911 | 0.9345 | never | stuck in local optimum | ✅ |
| lr 0.001 | 0.99993 | 0.9906 | 1,895 | converges | ✅ |
| lr 0.0001 | 0.98892 | 0.9761 | 8,901 | slow | ✅ |
| lr 0.00001 | 0.97069 | 0.9653 | never | stuck | ✅ |

Panels (c) and (d) match closely. In (a) and (b), all settings converge to
≈ 1 and the differences are small; the specific effects the paper
describes (memory 128 fluctuating, memory 2048 clearly slower) are not
visible with this seed. One seed per setting cannot separate such small
effects from randomness.

## Fig. 10 - quantization method (`figures/fig10_quantization.png`)

| K | Order-preserving: final MA | KNN: final MA | Paper |
|---|---|---|---|
| 10 | 0.9997 (≥ 0.99 at t = 1,119) | 0.9998 (≥ 0.99 at t = 1,261) | both → 1; OP faster |
| 2 | 0.9784 | **0.9784 (identical)** | both ~0.95; OP faster |
| 1 | 0.8332 | **0.8332 (identical)** | neither converges |

- K = 1: ✅ neither converges.
- K = 10: ✅ both reach ≈ 1; OP reaches 0.99 slightly earlier, though KNN is
  ahead around t = 2,000, so "OP converges faster" is only weakly supported.
- **K = 2: the two methods are mathematically identical.** Both produce
  the rounded action plus the action that flips the most uncertain device
  (the second-nearest binary vector to x̂ is exactly that flip). With the
  same seed our two runs are bit-for-bit equal. The gap the paper shows
  for K = 2 must therefore come from randomness between separate runs, not
  from the method. Our K = 2 level (0.978) is also higher than the paper's
  (~0.95).

## Figs. 11-13 - updating interval Δ for adaptive K

| Δ | final MA of Q̂ | mean K (last 2,000 frames) | CPU time, 10,000 frames |
|---|---|---|---|
| 1 | 0.9796 | 2.2 | 83 s |
| 2 | 0.9801 | 2.2 | 84 s |
| 4 | 0.9834 | 2.9 | 104 s |
| 8 | 0.9985 | 3.5 | 136 s |
| 16 | 0.9996 | 4.1 | 171 s |
| 32 | 0.9994 | 4.5 | 202 s |
| 64 | 0.9997 | 5.3 | 223 s |
| 128 | 0.9998 | 5.5 | 256 s |
| 256 | 0.9996 | 5.6 | 285 s |
| 512 | 0.9998 | 6.7 | 309 s |
| 1024 | 0.9997 | 6.5 | 332 s |
| ∞ (K = 10) | 0.9997 | 10 | 410 s |

- Fig. 11 ✅: small Δ (1, 2, 4) stays below optimal (~0.98; paper
  0.96-0.985); Δ ≥ 16 converges to ≈ 1 (paper: "Δ ≥ 16").
- Fig. 12 ✅: K_t grows with Δ.
- Fig. 13 ✅: the rate rises steeply up to Δ = 8-16 then flattens, while CPU
  time keeps increasing (paper: 0.96 → ~1, 68 → 230 s; ours 0.98 → ~1,
  83 → 332 s). Δ = 32 is a reasonable trade-off, as the paper chose.
  (CPU times measured with other runs in parallel; see Table 2 note.)

## Fig. 14 - comparison with benchmark methods (`figures/fig14_methods.png`)

Average weighted sum computation rate on the 6,000 test channels, in
10^6 bits/s (paper values read off its bar chart, approximate):

| N | CD | DROO | LR | Edge | Local |
|---|---|---|---|---|---|
| 10 | 3.19 (3.2) | 3.19 (3.2) | 3.16 (3.15) | 2.89 (2.9) | 0.94 (0.95) |
| 20 | 4.37 (4.4) | 4.35 (4.35) | 4.30 (4.3) | 3.69 (3.7) | 1.89 (1.9) |
| 30 | 6.03 (6.0) | 5.97 (6.0) | 5.90 (5.95) | 4.86 (4.85) | 2.86 (2.85) |

✅ Same ordering and essentially the same values for all five methods.

## Fig. 15 - normalized rate distribution (`figures/fig15_boxplot.png`)

Normalized against the optimum (N = 10) or CD (N = 20, 30).

| N | DROO median | DROO 25th pct | DROO ≥ 0.99 | LR median | LR 25th pct | Paper (median DROO / LR) |
|---|---|---|---|---|---|---|
| 10 | 1.0000 | 1.0000 | 99.2% | 0.9924 | 0.9864 | ~1.0 / ~0.997 |
| 20 | 1.0000 | 0.9947 | 86.4% | 0.9840 | 0.9765 | ~0.9995 / ~0.992 |
| 30 | 0.9980 | 0.9898 | 74.7% | 0.9796 | 0.9724 | ~0.999 / ~0.991 |

✅ DROO's median is ≈ 1 for all N and above LR's. DROO's spread for N = 30
is a little wider than in the paper's plot. DROO exceeds CD on a few
channels, as the paper notes.

**Data finding:** for N = 10, DROO "beat the optimum" on 0.1% of
channels. Re-solving those channels exhaustively shows the stored optimum
in the authors' data file is slightly low there (by up to 0.1%), and DROO
had found the true optimum exactly.

**Our LR** is about 0.5-1% below the paper's LR. LR is not in the authors'
code, and the paper's description is not fully specified (the relaxed
problem as literally written is not convex); our convex version is
described in README.md.

## Table 2 - CPU execution latency per channel

Seconds per channel; paper values (Intel i5-4570) in brackets.

| N | DROO fixed K = N | DROO adaptive (Δ = 32) | CD | LR (solver set up once) | LR (set up per channel) |
|---|---|---|---|---|---|
| 10 | 0.040 (0.036) | 0.015 (0.012) | 0.22 (0.20) | 0.0086 (0.24) | 0.14 |
| 20 | 0.16 (0.13) | 0.038 (0.030) | 1.4 (1.3) | 0.014 (0.53) | 0.28 |
| 30 | 0.39 (0.31) | 0.061 (0.059) | 4.5 (3.8) | 0.018 (0.81) | 0.43 |

Times relative to DROO (adaptive), ours (paper):

| N | fixed K | CD | LR (per channel) |
|---|---|---|---|
| 10 | 2.6× (3.0×) | 14× (17×) | 8.9× (20×) |
| 20 | 4.2× (4.3×) | 38× (43×) | 7.3× (18×) |
| 30 | 6.3× (5.3×) | **73× (65×)** | 7.0× (14×) |

- ✅ Adaptive K is much faster than fixed K, and DROO is 1-2 orders of
  magnitude faster than CD - the paper's main latency claim.
- ✅ Our absolute times are close to the paper's (within ~30%). Note these
  were measured with 11 runs in parallel, which slowed each run on this
  machine (e.g. fixed-K DROO for N = 10: 0.030 s per frame alone vs. 0.040 s
  in parallel); ratios between methods are the fairer comparison.
- ❌ **DROO vs. LR depends on how LR is implemented.** When the convex
  problem is set up once and only re-solved per channel (modern CVXPY with
  parameters), LR takes ~0.01-0.02 s and is *faster* than DROO. Setting it
  up for every channel - which the paper's times suggest - makes LR 7-9×
  slower than DROO (paper 14-20×). DROO still gives better rates than LR
  (Fig. 14-15).
- Fixed-K DROO was timed over 5,000 frames instead of 30,000 (its per-frame
  cost does not shrink over time).

**Where the time goes:** profiling the original `mainPyTorch.py` showed
~89% of the run time in the (P2) bisection solver (41.8 million `lambertw`
calls), ~5% in the network's forward pass and ~1-2% in training,
confirming the paper's remark that DROO's complexity "greatly depends on"
solving (P2).

---

## Bonus (not in the paper): does a GPU help?

`experiments/gpu_benchmark.py`, run in a separate environment with
CUDA PyTorch (`.venv-gpu`) on an RTX 5070 Ti Laptop GPU. Same DROO run
(adaptive K, Delta = 32, seed 0, 3,000 frames) on CPU and GPU. The network
is the authors' MemoryDNN, moved to the GPU by a subclass (checked to give
bit-identical results to the original on the CPU).

| N | Device | Total s/frame | Network s/frame | Solver + rest s/frame | Avg normalized rate |
|---|---|---|---|---|---|
| 10 | CPU | 0.0212 | 0.00078 | 0.0204 | 0.99370 |
| 10 | GPU | 0.0204 | 0.00195 | 0.0184 | 0.99349 |
| 30 | CPU | 0.1885 | 0.00107 | 0.1875 | 0.95261 |
| 30 | GPU | 0.2058 | 0.00295 | 0.2029 | 0.95750 |

- **The GPU makes the network 2.5-2.8x slower**, not faster. The network is
  tiny (two hidden layers, 200 neurons) and is called once per frame with a
  single input, so copying data to and from the GPU and launching GPU work
  costs more than the computation itself.
- **The network is a small part of the time anyway:** ~4% of a frame for
  N = 10 and ~0.6% for N = 30 on the CPU. The rest is the (P2) bisection
  solver, which is plain Python/SciPy and cannot use the GPU.
- The whole-run differences (+4% for N = 10, -8% for N = 30) come from the
  CPU-side "solver + rest" part, not the network. GPU arithmetic rounds
  slightly differently, so the two runs make slightly different decisions
  (and adaptive K evaluates a different number of candidates) - which also
  explains the small differences in the rate. These are run-to-run
  variation, not a GPU speed-up.
- **Conclusion:** for DROO as published, a GPU brings no benefit; the CPU
  is the right choice, as in the paper. A GPU would only pay off with a much
  larger network or many channels processed in one batch.

---

## Differences from the paper, collected

1. **Training loss** ~0.10 instead of ~0.04 (PyTorch re-creates Adam each
   step; beta1 = 0.09). Rates unaffected.
2. **Demos use adaptive K**, while the captions of Fig. 6 and 8 say K = 10;
   we ran both. Adaptive K misses Fig. 8's 0.99 threshold.
3. **Fig. 8:** devices are switched off in a fixed order, not randomly.
4. **Fig. 10:** for K ≤ 2, OP and KNN are identical by construction.
5. **Fig. 7:** WD1's gain +37% vs. +26% reported.
6. **LR:** our own implementation (not in the authors' code); slightly
   lower rates and much faster than the paper's timing when the solver is
   reused.
7. **Stored N = 10 optimum** is slightly below the true optimum on ~0.1% of
   test channels.
8. **One seed per configuration.** Small differences (e.g. Fig. 9(a), (b))
   could change with other seeds.

## How to reproduce

See README.md. In short: `python experiments/fig05.py`, `fig06.py`,
`fig08.py` (with and without `--adaptive`), then
`python experiments/run_all.py` (~35 min on a Ryzen 9 with 11 parallel
runs).
