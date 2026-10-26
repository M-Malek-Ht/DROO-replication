"""
p2_vectorized.py - the authors' resource-allocation solver for (P2), rewritten
to solve MANY offloading actions at once.

The authors' bisection() in DROO/optimization.py solves (P2) for ONE action
x with a bisection search over the dual variable v. Here the exact same
equations, constants and stopping rule are applied to a whole matrix of
actions in one go with numpy. Because the bisection always halves the same
interval [0, 999999999] until it is narrower than 0.005, every action takes
the same number of steps, so the results match bisection() to floating-point
precision (checked in test_p2_vectorized.py).

Why we need it: some experiments must try ALL 2^N actions for a channel
(exhaustive search = the true optimum Q*, the denominator of the normalized
rate in eq. (11)). Calling bisection() 1,024 times per frame is too slow for
thousands of frames; this version does it in milliseconds.

Also returns each WD's own (unweighted) computation rate, needed for Fig. 7.
"""

import itertools

import numpy as np
from scipy.special import lambertw

# Constants: identical to DROO/optimization.py bisection()
O = 100            # phi: cycles per bit
P = 3              # AP transmit power (W)
U = 0.7            # energy harvesting efficiency used in the code
ETA1 = ((U * P) ** (1.0 / 3)) / O
KI = 10 ** -26     # computation energy efficiency coefficient
ETA2 = U * P / 10 ** -10
B = 2 * 10 ** 6    # bandwidth
VU = 1.1
EPSILON = B / (VU * np.log(2))


def default_weights(N):
    """Authors' default weights: [1, 1.5, 1, 1.5, ...]."""
    return np.array([1.5 if i % 2 == 1 else 1.0 for i in range(N)])


def all_actions(N):
    """All 2^N binary offloading actions, one per row."""
    return np.array(list(itertools.product([0, 1], repeat=N)), dtype=float)


def solve_p2(h, M, weights=None):
    """
    Solve (P2) for every action (row) of M at once.

    h        channel gains, shape (N,), in real units (NOT multiplied by 1e6)
    M        binary actions, shape (K, N) or (N,)
    weights  WD weights, shape (N,); None = authors' default

    Returns a dict with arrays (one entry per action):
        Q      weighted sum computation rate  (same as bisection()[0])
        a      energy-transfer time fraction  (same as bisection()[1])
        tau    offloading time fractions, shape (K, N) (0 for local WDs)
        r_wd   each WD's own computation rate, shape (K, N) (unweighted)
    """
    h = np.asarray(h, dtype=float)
    M = np.atleast_2d(np.asarray(M, dtype=float))
    N = h.shape[0]
    w = default_weights(N) if weights is None or len(weights) == 0 \
        else np.asarray(weights, dtype=float)

    off = M == 1                      # offloading WDs (set M1 in the code)
    loc = ~off                        # local WDs (set M0)
    h2 = h ** 2
    local_coef = w * ETA1 * (h / KI) ** (1.0 / 3)          # per WD
    sum_local_coef = (loc * local_coef).sum(axis=1)        # per action

    def phi(v):
        # v: shape (K,) -> phi: shape (K, N); only used where off is True
        z = -1.0 / np.exp(1 + v[:, None] / w[None, :] / EPSILON)
        return 1.0 / (-1 - 1.0 / lambertw(z).real)

    def p1(ph):
        return 1.0 / (1 + (off * h2[None, :] * ph).sum(axis=1) * ETA2)

    def Q(v):
        ph = phi(v)
        s1 = sum_local_coef * p1(ph) ** (-2.0 / 3) / 3
        s2 = (off * w[None, :] * h2[None, :] / (1 + 1.0 / ph)).sum(axis=1)
        return s1 + s2 * EPSILON * ETA2 - v

    # Bisection, same start, same stopping rule as the authors' loop
    K = M.shape[0]
    UB = np.full(K, 999999999.0)
    LB = np.zeros(K)
    v = np.zeros(K)
    delta = 0.005
    while UB[0] - LB[0] > delta:          # same width for every action
        v = (UB + LB) / 2
        pos = Q(v) > 0
        LB = np.where(pos, v, LB)
        UB = np.where(pos, UB, v)

    ph = phi(v)
    a = p1(ph)
    tau = np.where(off, ETA2 * h2[None, :] * a[:, None] * ph, 0.0)

    r_local = ETA1 * (h / KI) ** (1.0 / 3) * a[:, None] ** (1.0 / 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_off = EPSILON * tau * np.log(1 + ETA2 * h2[None, :] * a[:, None] / tau)
    r_wd = np.where(off, np.nan_to_num(r_off), r_local)

    Qval = (w[None, :] * r_wd).sum(axis=1)
    return {"Q": Qval, "a": a, "tau": tau, "r_wd": r_wd}


def optimum(h, weights=None, actions=None):
    """
    Exhaustive search over all 2^N actions (the true optimum of (P1)).
    Returns (best Q, best action, per-WD rates under the best action).
    """
    if actions is None:
        actions = all_actions(len(h))
    out = solve_p2(h, actions, weights)
    k = int(np.argmax(out["Q"]))
    return out["Q"][k], actions[k], out["r_wd"][k]
