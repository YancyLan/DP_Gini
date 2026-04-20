from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import time

from .utils import sort_X, cal_gini

def take_one_out(ind, x):
    x_new = np.delete(x, ind)
    return cal_gini(x_new)

def find_j_star(g, x, n, tol = 1e-7):
    '''
    Return the integer *j* in the closed interval ``[1, n]`` that minimises ``g(j)``.
    Parameters
    ------------
    g: Function to take_one_out or insert_one_in
    '''
    lo = 0
    hi = n-1

    # Discrete ternary search
    while hi - lo > 3:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3

        if g(m1,x) < g(m2,x):
            # Minimum lies in [lo, m2 − 1]
            hi = m2 - 1
        else:
            # Minimum lies in [m1 + 1, hi]
            lo = m1 + 1

    j_star = min(range(lo, hi + 1), key=lambda j: g(j, x))
    return j_star

def find_j_star_1d(g, lo, hi):
    """
    Discrete ternary search over integers j in [lo, hi], assuming g is unimodal.
    Returns argmin j.
    """
    while hi - lo > 3:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3
        if g(m1) < g(m2):
            hi = m2 - 1
        else:
            lo = m1 + 1
    best_j = min(range(lo, hi + 1), key=g)
    return best_j

def fast_min_gini(x, k):
    """
    O(k log(n-k)) accelerated minimum Gini using correct rank-weighted objective.
    x must be sorted ascending.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if not 0 < k < n:
        raise ValueError("k must be between 1 and n-1")

    # Prefix sums:
    # pref_x[t]  = sum_{p < t} x[p]
    # pref_px[t] = sum_{p < t} p * x[p]
    pref_x  = np.concatenate(([0.0], np.cumsum(x)))
    idx     = np.arange(n, dtype=float)
    pref_px = np.concatenate(([0.0], np.cumsum(idx * x)))

    def sum_x(L, R):    # sum over [L, R)
        return pref_x[R] - pref_x[L]

    def sum_px(L, R):   # sum over [L, R) of p*x[p]
        return pref_px[R] - pref_px[L]

    best = np.inf

    # Loop i = number taken from the *lower* tail (thus k-i from upper tail)
    for i in range(k + 1):
        L = i
        R = n - k + i  # exclusive; middle window is x[L:R]
        # Precompute den parts that don't depend on j: total sum of y
        # y = [x[L:j], k copies of v=x[j], x[j:R]]
        # sum(y) = sum_x(L,R) + (k-1)*v, but easier to compute per j

        def g_of_j(j_rel):
            j = L + j_rel                 # absolute index
            v = x[j]
            a = j - L                     # len of left-of-v block
            # A = x[L:j] placed at positions t=1..a
            sumA = sum_x(L, j)
            sum_rA_x = sum_px(L, j) - L * sumA            # r = 0..a-1
            sum_tA_x = sum_rA_x + sumA                    # t = r+1
            contrib_A = 2 * sum_tA_x - (n + 1) * sumA

            # V = k copies of v at positions t = a+1 .. a+k
            # sum_{t=a+1}^{a+k} (2t - n - 1) = k*(2a + k - n)
            contrib_V = k * v * (2 * a + k - n)

            # B = x[j:R] placed at positions t = a+k+1 .. a+k+b
            sumB = sum_x(j, R)
            C = 2 * (a + k) - n - 1
            sum_rB_x = sum_px(j, R) - j * sumB            # r' = 0..b-1
            sum_uB_x = sum_rB_x + sumB                    # u = r'+1
            contrib_B = 2 * sum_uB_x + C * sumB

            num = contrib_A + contrib_V + contrib_B
            den = n * (sumA + k * v + sumB)
            return num / den

        # minimize g_of_j over j ∈ [L, R-1] i.e. j_rel ∈ [0, R-L-1]
        if R > L:
            j_star_rel = find_j_star_1d(g_of_j, 0, R - L - 1)
            best = min(best, g_of_j(j_star_rel))

    return best


def fast_max_gini(x_sorted, k, L, U):
    n = len(x_sorted)
    if not 0 < k < n:
        raise ValueError("k must be between 1 and n-1")

    pref = np.concatenate(([0.0], np.cumsum(x_sorted, dtype=float)))
    coef = 2 * np.arange(1, n + 1) - n - 1
    coeff_pref = np.concatenate(([0.0], np.cumsum(coef * x_sorted, dtype=float)))

    best = -1.0
    for i in range(n - k + 1):
        left_sum = pref[i]
        right_sum = pref[n] - pref[i + k]

        for j in range(k + 1):
            nL, nU = j, k - j
            new_sum = left_sum + right_sum + nL * L + nU * U
            num = (
                L * nL * (nL - n)
                + coeff_pref[i] + 2 * nL * left_sum
                + (coeff_pref[n] - coeff_pref[i + k]) - 2 * nU * right_sum
                + U * nU * (n - nU)
            )
            g = num / (n * new_sum)
            if g > best:
                best = g
    return best

## calculate max average
def cal_max_ave(x, k, U):
    x[0:k] = U
    return np.mean(x)

## calculate min average
def cal_min_ave(x, k, L):
    x[-k:] = L
    return np.mean(x)

# calculate beta
def cal_beta(eps, gamma =2.5):
    # alpha = eps/(4*gamma)
    beta = eps/gamma
    return beta

# calculate smooth upper bound
def smooth_upper_bound(beta, x, L, U) -> float:
    n        = x.size
    if n < 2:
        raise ValueError("Need at least two observations")

    # L, U     = x[0], x[-1]     # min and max
    ave      = x.mean()
    g        = cal_gini(x)

    if n * ave - (U - L) > 0 :
        cand_1 = max((U - L) * (1 - g) / (n * ave + (U -  L)),
                 2 * (ave - L) / (n * ave))

        cand_2 = max(((U - L) * (g + 1)) / (n * ave - (U - L)),
                 2 * (U - ave) / (n * ave - (U - L)))
        cand = max(cand_1, cand_2)
    else:
        cand = 1
    return math.exp(-beta*1)*min(cand, 1)

# Calibrate Smooth Upper Bound
def cal_su(x, beta, L, U):
    
    best = 0
    # calculate the local sensitivity of the Gini index
    ls_ori = smooth_upper_bound(beta=beta, x=x, L=L, U=U)
    best = ls_ori
    print("ls_ori",ls_ori)
    print("beta",beta)
    n = len(x)
    k_m = -math.log(ls_ori)/beta
    k_m = math.ceil(min(n,k_m))
    print(k_m)

    # record time in each run
    for k in range(1, k_m):
        t0 = time.time()
        g_M = fast_max_gini(x, k, L, U)
        g_m = fast_min_gini(x, k)
        ave_M = cal_max_ave(x, k, U)
        ave_m = cal_min_ave(x, k, L)

        cand_1 = max((U-L)*(1-g_m)/(n*ave_m + (U-L)) , 2*(ave_M-L)/(n*ave_m))
        cand_2 = max(((U-L)*(g_M+1))/(n*ave_m - (U-L)), 2*(U-ave_m)/(n*ave_m - (U-L)))
 
        cand = max([cand_1, cand_2])
        cand = min(cand,1)
        su_k = math.exp(-beta*k)*min(cand, 1)
        best = max(su_k, best)
        t1 = time.time()
        print(f"Time for k={k}: {t1 - t0:.6f} seconds")
        print(k, "smooth_upper_bound:", best)
    return best, k_m

def cal_su_fast(x, beta, L, U, xm = None):
    beta = float(np.squeeze(beta))
    L    = float(np.squeeze(L))
    U    = float(np.squeeze(U))
    if xm is None:
        xm   = float(x.mean())

    # To ensure the robust of our model
    eta = 1e-12
    # L_safe must be positive for the ratio check to work correctly
    L_safe = max(L + eta, eta)
    n = len(x)
    iq = (U - L) / xm

    # calculate the local sensitivity of the Gini index
    ls_ori = 2/(n*(1/iq)-1)
    ls_ori = float(ls_ori)
    best = ls_ori
    # print("ls_ori",ls_ori)
    # rint("beta",beta)
    if ls_ori <= 0:
        return 0
    k_m = -math.log(ls_ori)/beta
    k_m = math.ceil(min(n,k_m))
    # print(k_m)

    for k in range(1, k_m):
        ak = 2/(n/(iq-k/n)-1)
        if 1/(iq-k/n) < (U-L)/L_safe and ak<1:
            su_k =  math.exp(-beta*k)*min(ak, 1)
            best = max(best, su_k)
            # print(k, "smooth_upper_bound:", best)
        else:
            ak = 1
            su_k =  math.exp(-beta*k)*min(ak, 1)
            best = max(best, su_k)
            break
    return best

# Calibrate Noise
def sample_eta(gamma: float, size: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    if gamma <= 1:
        raise ValueError("gamma must be > 1")
    rng = rng or np.random.default_rng()
    r = rng.random(size)
    magnitude = np.power(r, -1.0 / (gamma - 1.0)) - 1.0
    signs = rng.choice([-1, 1], size=size)
    return signs * magnitude
def add_dp_noise_gini(gini_true: float, S_value: float, epsilon: float, gamma: float,
                      rng: np.random.Generator | None = None) -> float:
    eta = sample_eta(gamma, 1, rng=rng)[0]
    return gini_true + 2 * (gamma + 1) / epsilon * S_value * eta
def simulate_noisy_gini(gini_true: float, S_value: float, epsilon: float, gamma: float =2,
                        n_samples: int = 10000, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    eta = sample_eta(gamma, n_samples, rng=rng)
    noise = 4 * gamma / epsilon * S_value * eta
    return gini_true + noise


## Global sensitivity
def sample_exponential_noise(eps, size, rng=None):
    if eps <= 0:
        raise ValueError("ε must be > 0")
    rng = rng or np.random.default_rng()
    exp_draw = rng.exponential(scale=1 / eps, size=size)
    signs = rng.choice([-1, 1], size=size)
    return signs * exp_draw

# Algorithm 1: AboveThreshold
def laplace(scale):
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))

def above_threshold(data, queries, T, Delta, eps1, eps2):
    # 1. noisy threshold
    T_hat = T + laplace(Delta / eps1)

    # 2. iterate every query
    for i, f in enumerate(queries):
        v_i = laplace(Delta / eps2)            # add noise to the query result
        if f(data) + v_i >= T_hat:             # if exceeds noisy threshold
            print(f"Query {i}: ⊤ (halt)")      # output True and halt
            return i
        else:
            print(f"Query {i}: ⊥")             # output False
    return None

def unbounded_quantile_mech(x, q, ell, beta, eps1=0.5, eps2=0.5, i_max=None, seed=None):
    """
    x: data list
    q: quantile (0~1), e.g., median 0.5
    ell: known/assumed lower bound ℓ
    beta: β > 1 (exponential growth factor)
    eps1, eps2: privacy budgets for AboveThreshold
    i_max: number of queries to generate (if None, automatically cover up to max(x))
    """
    if seed is not None:
        random.seed(seed)

    n = len(x)
    T = q * n              # count: qn when q=1, count = n
    Delta = 1.0            # global sensitivity of counting query is 1

    # Automatically determine the number of queries needed (until the right endpoint exceeds max(x))
    if i_max is None:
        xmax = max(x)
        # Need β^i + ℓ - 1 to cover xmax, find the minimum i
        target = max(1.0, xmax - ell + 1)
        i_max = max(1, math.ceil(math.log(target, beta)))

    # f_i(x) = |{x_j ∈ x | x_j - ℓ + 1 < β^i}| = count of x_j < β^i + ℓ - 1
    cutoffs = [beta**i + ell - 1 for i in range(i_max)]
    queries = [lambda d, c=c: sum(1 for v in d if v < c) for c in cutoffs]

    k = above_threshold(x, queries, T, Delta, eps1, eps2)
    if k is None:
        k = i_max - 1
    return (beta**k + ell - 1), k, cutoffs

# Private upper bound
def laplace(scale):
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))

def above_threshold(data, queries, T, Delta, eps1, eps2):
    # 1. noisy threshold
    T_hat = T + laplace(Delta / eps1)

    # 2. iterate every query
    for i, f in enumerate(queries):
        v_i = laplace(Delta / eps2)            # add noise to the query result
        if f(data) + v_i >= T_hat:             # if exceeds noisy threshold
            return i
    return None

def unbounded_quantile_mech(x, q, ell, beta, eps1=0.5, eps2=0.5, i_max=None, seed=None):
    """
    x: data list
    q: quantile (0~1), e.g., median 0.5
    ell: known/assumed lower bound ℓ
    beta: β > 1 (exponential growth factor)
    eps1, eps2: privacy budgets for AboveThreshold
    i_max: number of queries to generate (if None, automatically cover up to max(x))
    """
    if seed is not None:
        random.seed(seed)

    n = len(x)
    T = q * n              # count: qn when q=1, count = n

    Delta = 1.0            # global sensitivity of counting query is 1

    # Automatically determine the number of queries needed (until the right endpoint exceeds max(x))
    if i_max is None:
        xmax = max(x)
        # Need β^i + ℓ - 1 to cover xmax, find the minimum i
        target = max(1.0, xmax - ell + 1)
        i_max = max(1, math.ceil(math.log(target, beta)))

    # f_i(x) = |{x_j ∈ x | x_j - ℓ + 1 < β^i}| = count of x_j < β^i + ℓ - 1
    cutoffs = [beta**i + ell - 1 for i in range(i_max)]
    queries = [lambda d, c=c: sum(1 for v in d if v < c) for c in cutoffs]

    k = above_threshold(x, queries, T, Delta, eps1, eps2)
    if k is None:
        k = i_max - 1
    return (beta**k + ell - 1), k, cutoffs


if __name__ == "__main__":
    x = np.array([10,20,30,40,50,60,70,80,90])
    x = sort_X(x)
    L = x[0]
    U = x[-1]
    eps = 1.0
    beta = cal_beta(eps=eps, gamma=2.5)
    su = cal_su(x, beta, L, U)
    print("Final Smooth Upper Bound:", su)
