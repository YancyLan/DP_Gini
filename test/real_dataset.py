from pathlib import Path
from random import seed
import sys

ROOT = Path(__file__).resolve().parents[1]   # …/DP_GINI
sys.path.insert(0, str(ROOT / "src"))

from DPGini.utils import cal_gini, cal_gini, sort_X

import DPGini as dg
from DPGini.core import (
    cal_beta, smooth_upper_bound, cal_su_fast, sample_eta, add_dp_noise_gini, simulate_noisy_gini, laplace, above_threshold, unbounded_quantile_mech
)
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from matplotlib.patches import Patch
from pandas import DataFrame


if __name__ == "__main__":
    path = ROOT / "data" / "asecpub24csv"/ f"pppub24.csv"
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.upper() for c in df.columns]
    cols = [c for c in ["PH_SEQ","PPPOS","A_AGE","MARSUPWT","PTOTVAL"] if c in df.columns]
    d = df[cols].copy()
    del df

    # Coerce numerics
    for c in ["A_AGE","MARSUPWT","PTOTVAL"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")


    # Sensible filters:
    # - Age 15+ (income universe)
    # - Positive person weight (represents population; 0-weight rows add nothing)
    # - PTOTVAL not missing (keep 0 = no income; keep negatives = losses)
    mask = (
        (d["A_AGE"] >= 15) &
        (d["MARSUPWT"] > 0) &
        (d["PTOTVAL"].notna())
    )
    df_clean = d[mask].copy()
    df_clean = df_clean[df_clean["PTOTVAL"] >= 0]

    s = pd.to_numeric(df_clean["PTOTVAL"], errors="coerce")
    min_all = s.min()
    max_all = s.max()
    mean_all = s.mean()
    X = np.sort(s.to_numpy(dtype=np.float64, copy=False))
    g_real = cal_gini(X)
    print("PTOTVAL min:", min_all)
    print("PTOTVAL max:", max_all)
    print("PTOTVAL mean:", mean_all)
    print("Gini of the real dataset", g_real)
    rate = max_all / mean_all
    print("Max/mean ratio:", rate)

    eps_list = [0.25, 0.5, 1, 1.5, 2]
    beta_list = [cal_beta(eps=i, gamma=2) for i in eps_list] 
    gamma = 2
    n_samples = 100_000
    L = 0
    rng = np.random.default_rng(seed=42)

    # for every eps we generate 100000 samples 
    df = pd.DataFrame(index=range(n_samples), columns=eps_list, dtype=float)

    for e, b in zip(eps_list, beta_list):
        for n in range(n_samples):
            U = 2.5 * unbounded_quantile_mech(
                x=X, q=1, ell=min_all, beta=1.5, eps1=0.75, eps2=0.75
            )[0]
            su = cal_su_fast(X, beta=b, L=L, U=U)
            noisy_gini = simulate_noisy_gini(
                gini_true=g_real,
                S_value=su,
                epsilon=e,
                gamma=gamma,
                n_samples=1,   
                rng=rng
            )
            df.at[n, e] = float(noisy_gini)
            print(n)

    

    cols = sorted(df.columns, key=float)

    # plot
    data = [df[c].dropna().astype(float).to_numpy() for c in cols]

    plt.figure(figsize=(9, 4))
    plt.boxplot(
        data,
        widths=0.6,
        medianprops=dict(color='lightblue', linewidth=2),
        showfliers=False,   
    )

    plt.xticks(range(1, len(cols) + 1), [str(c) for c in cols])
    plt.xlabel(r'$\varepsilon$')
    plt.ylabel('Private Gini estimate')
    plt.tight_layout()

    plt.show()


