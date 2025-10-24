from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]   # …/DP_GINI
sys.path.insert(0, str(ROOT / "src"))

import DPGini as dg
from DPGini.core import (
    take_one_out, find_j_star, find_j_star_1d, cal_max_ave, cal_min_ave
)
import numpy as np
import math

def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol

def run():
    # ----- basic data -----
    x_eq = dg.sort_X(np.array([5, 5, 5, 5], dtype=float))
    x    = dg.sort_X(np.array([1, 2, 3, 4], dtype=float))
    x5   = dg.sort_X(np.array([1, 2, 3, 4, 5], dtype=float))
    L, U = float(x5[0]), float(x5[-1])

    # ----- gini -----
    assert approx(dg.cal_gini(x_eq), 0.0)

    # ----- beta -----
    assert approx(dg.cal_beta(1.0, 2.5), 0.4)

    # ----- fast min/max gini -----
    gmin = dg.fast_min_gini(x5, k=1)
    gmax = dg.fast_max_gini(x5, k=1, L=L, U=U)
    assert 0.0 <= gmin <= 1.0
    assert 0.0 <= gmax <= 1.0

    # ----- smooth upper bound (fast) -----
    beta = dg.cal_beta(1.0, 2.5)
    su_f = dg.cal_su_fast(x5, beta, L, U)
    assert 0.0 <= su_f <= 1.0

    # (optional slower exact) su
    su_exact = dg.cal_su(x5.copy(), beta, L, U)
    assert 0.0 <= su_exact <= 1.0

    # ----- DP noise -----
    g_true = float(dg.cal_gini(x5))
    rng = np.random.default_rng(0)
    g_dp = dg.add_dp_noise_gini(g_true, su_f, epsilon=1.0, gamma=2.5, rng=rng)
    assert isinstance(g_dp, float) and math.isfinite(g_dp)

    # simulate_noisy_gini shape
    samples = dg.simulate_noisy_gini(g_true, su_f, epsilon=1.0, gamma=2.0, n_samples=128)
    assert samples.shape == (128,)

    # sample_eta shape and signs
    eta = dg.sample_eta(gamma=2.2, size=256)
    assert eta.shape == (256,)
    assert np.any(eta > 0) and np.any(eta < 0)

    # symmetric exponential noise
    en = dg.sample_exponential_noise(eps=1.5, size=512)
    assert en.shape == (512,)
    assert np.any(en > 0) and np.any(en < 0)

    # ----- helper funcs -----
    # take_one_out returns a valid gini
    g_take = take_one_out(0, x5)
    assert 0.0 <= g_take <= 1.0

    # find_j_star_1d on a convex (unimodal) function
    argmin = find_j_star_1d(lambda j: (j - 5) ** 2, 0, 10)
    assert argmin == 5

    # find_j_star with g(j, x)
    arr = np.array([3, 1, 0, 2], dtype=float)
    j_star = find_j_star(lambda j, a: a[j], arr, n=len(arr))
    assert j_star == 2  # index of minimum (value 0)

    # averages helpers
    base_mean = float(np.mean(x5))
    assert cal_max_ave(x5.copy(), k=2, U=U) >= base_mean
    assert cal_min_ave(x5.copy(), k=2, L=L) <= base_mean

    print("All checks passed ✅")

if __name__ == "__main__":
    run()