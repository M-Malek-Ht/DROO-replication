"""
test_p2_vectorized.py - checks that our vectorized solver gives the SAME
answers as the authors' bisection() (DROO/optimization.py).

Run from the project folder:
    python baselines/test_p2_vectorized.py

It checks:
  1) random actions, default and custom weights, N = 10, 20, 30
  2) exhaustive search on N = 10 reproduces the stored optimum (output_obj)
     of the authors' data file for the first 200 channels
Prints PASS/FAIL for each check.
"""

import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DROO"))
sys.path.insert(0, str(ROOT / "baselines"))

from optimization import bisection                    # noqa: E402
from p2_vectorized import optimum, solve_p2           # noqa: E402

rng = np.random.default_rng(0)
ok = True

# 1) random actions vs the authors' solver
for N in (10, 20, 30):
    h_all = sio.loadmat(str(ROOT / "DROO" / "data" / ("data_%d" % N)))["input_h"]
    worst = 0.0
    for trial in range(40):
        h = h_all[trial]
        M = rng.integers(0, 2, size=(5, N))
        w = list(rng.choice([1, 1.5, 3], size=N)) if trial % 2 else []
        ours = solve_p2(h, M, w)
        for k in range(5):
            Qa, aa, ta = bisection(h, M[k], w)
            worst = max(worst, abs(ours["Q"][k] - Qa) / Qa,
                        abs(ours["a"][k] - aa) / max(aa, 1e-12))
    passed = worst < 1e-6
    ok &= passed
    print("N=%d  random actions: largest relative difference %.2e  %s"
          % (N, worst, "PASS" if passed else "FAIL"))

# 2) exhaustive search reproduces the stored optimum for N = 10
d = sio.loadmat(str(ROOT / "DROO" / "data" / "data_10"))
t = time.time()
ratios = [optimum(d["input_h"][i])[0] / d["output_obj"][i, 0] for i in range(200)]
dt = (time.time() - t) / 200
worst = max(abs(r - 1) for r in ratios)
passed = worst < 1e-6
ok &= passed
print("N=10 exhaustive search vs stored optimum: largest difference %.2e  %s"
      % (worst, "PASS" if passed else "FAIL"))
print("     (%.1f ms per channel for all 1,024 actions)" % (1000 * dt))

print()
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
sys.exit(0 if ok else 1)
