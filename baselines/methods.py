"""
methods.py - the benchmark methods the paper compares DROO against
(Section 5.3, Fig. 14, Fig. 15, Table 2).

  coordinate_descent(h)   CD  - the AUTHORS' code, DROO/optimization.py
                                cd_method(), used unchanged
  edge_computing(h)       all WDs offload  (x = 1, ..., 1), solved with the
                                authors' bisection()
  local_computing(h)      all WDs compute locally (x = 0, ..., 0), same
  linear_relaxation(h)    LR  - NOT in the authors' repository; written here
                                from the paper's description (see below)

Every method returns (Q, x): the weighted sum computation rate and the
binary offloading action it chose.

------------------------------------------------------------------------
About LR (our implementation - the paper gives only a short description)
------------------------------------------------------------------------
Paper (Sec. 5.3): "The binary offloading decision variable x_i conditioned
on (4d) is relaxed to a real number between 0 and 1, x_i in [0, 1]. Then the
optimization problem (P1) with this relaxed constraint is convex with
respect to {x_i} and can be solved using the CVXPY convex optimization
toolbox. Once x_i is obtained, the binary offloading decision x_i is
determined as  x_i = 1 if r*_O,i(a, tau_i) >= r*_L,i(a), else 0.   (12)"

Written literally, (P1) with x_i in [0, 1] is NOT jointly convex (it has
products such as (1 - x_i) * a^(1/3)). So we use the standard convex form
that equals (P1) whenever x is binary:
    local part :  w_i * eta1 * (h_i/k)^(1/3) * (1 - x_i)^(2/3) * a^(1/3)
                  (a geometric mean of (1 - x_i) and a, which is concave)
    offload part: w_i * eps * tau_i * log(1 + eta2 * h_i^2 * a / tau_i)
                  with the extra constraint tau_i <= x_i (an offloading
                  time can only be used by an offloading WD)
    constraints : sum(tau) + a <= 1, a >= 0, tau >= 0, 0 <= x <= 1
For x in {0, 1} this is exactly (P1). We solve it with CVXPY, apply rule
(12) with the relaxed solution's a and tau_i, and then evaluate the
resulting binary x with the authors' bisection() solver, like every other
method. This interpretation is recorded in the README and REPORT.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DROO"))
sys.path.insert(0, str(ROOT / "baselines"))

from optimization import bisection, cd_method           # noqa: E402 (authors')
from p2_vectorized import (EPSILON, ETA1, ETA2, KI,     # noqa: E402
                           default_weights)

SCALE = 1e-6   # rates are ~1e5..1e6; scaling helps the convex solver


def edge_computing(h):
    x = np.ones(len(h), dtype=int)
    return bisection(h, x)[0], x


def local_computing(h):
    x = np.zeros(len(h), dtype=int)
    return bisection(h, x)[0], x


def coordinate_descent(h):
    """The authors' CD method (random starting action, then greedy swaps)."""
    Q, x = cd_method(h)
    return Q, np.asarray(x)


class LinearRelaxation:
    """
    LR solver. The convex problem is built ONCE per N with CVXPY parameters
    (so it is not re-compiled for every channel), then re-solved per channel.
    """

    def __init__(self, N, weights=None):
        import cvxpy as cp
        self.cp = cp
        self.N = N
        self.w = default_weights(N) if weights is None else np.asarray(weights)

        self.c_loc = cp.Parameter(N, nonneg=True)   # w*eta1*(h/k)^(1/3)*SCALE
        self.c_snr = cp.Parameter(N, nonneg=True)   # eta2*h^2
        self.a = cp.Variable(nonneg=True)
        self.tau = cp.Variable(N, nonneg=True)
        self.x = cp.Variable(N)

        terms = []
        for i in range(N):
            local = self.c_loc[i] * cp.geo_mean(
                cp.hstack([1 - self.x[i], self.a]), [2, 1])
            offload = (self.w[i] * EPSILON * SCALE) * (
                -cp.rel_entr(self.tau[i], self.tau[i] + self.c_snr[i] * self.a))
            terms += [local, offload]
        cons = [cp.sum(self.tau) + self.a <= 1, self.x >= 0, self.x <= 1,
                self.tau <= self.x]
        self.problem = cp.Problem(cp.Maximize(cp.sum(cp.hstack(terms))), cons)

    def __call__(self, h):
        h = np.asarray(h, dtype=float)
        self.c_loc.value = self.w * ETA1 * (h / KI) ** (1.0 / 3) * SCALE
        self.c_snr.value = ETA2 * h ** 2
        try:
            self.problem.solve(solver=self.cp.CLARABEL)
            a = max(float(self.a.value), 1e-12)
            tau = np.maximum(np.asarray(self.tau.value).ravel(), 0)
        except Exception:                       # solver failure: fall back
            a, tau = 1.0, np.zeros(self.N)

        # Rule (12): offload if the offloading rate >= the local rate
        r_loc = ETA1 * (h / KI) ** (1.0 / 3) * a ** (1.0 / 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            r_off = np.where(tau > 0,
                             EPSILON * tau * np.log(1 + ETA2 * h ** 2 * a / tau), 0)
        x = (r_off >= r_loc).astype(int)
        return bisection(h, x)[0], x
