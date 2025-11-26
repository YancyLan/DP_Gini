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

def plot_abs_error_violin_grouped_by_n_then_eps(
    all_array, ci=0.95,
    eps_step=0.85, alpha_gap=0.35, width=0.16,
    x_tick_labelsize=14, y_tick_labelsize=14,
    axis_label_fontsize=14, legend_fontsize=14, title_fontsize=14,
    group_label_fontsize=14, label_fontsize=10,
  
    base_pad_dex=0.01,   
    rel_scale=0.3,      
    min_gap_dex=0.2,   
    pixel_bump_pt=2,     
    jitter_x_pt=2,       
    add_label_bbox=False
):
    eps_list   = [0.5, 1, 2]
    alpha_list = [1e4, 1e5, 1e6]
    eps_idx   = {0.5: 0, 1: 1, 2: 2}
    alpha_idx = {1e4: 0, 1e5: 1, 1e6: 2}

    g_list = [0.2, 0.5, 0.7]
    keys_sorted = [f"samples_laplace_{int(round(g*100)):02d}" for g in g_list]
    g_colors = {g: c for g, c in zip(g_list, ['C0', 'C1', 'C2'])}

    inner_offsets = np.linspace(-0.18, 0.18, num=len(g_list))

    n_alpha = len(alpha_list)
    n_eps   = len(eps_list)
    slots_per_alpha = n_eps

    slot_centers = []
    for a_i in range(n_alpha):
        block_start = a_i * (slots_per_alpha * eps_step + alpha_gap)
        for e_i in range(n_eps):
            slot_centers.append(block_start + e_i * eps_step)

    tail = (1.0 - ci) / 2.0
    q_hi = 1.0 - tail
    ymin = 1e-6

    data_by_g = {g: [] for g in g_list}   
    xpos_by_g = {g: [] for g in g_list}
    y99_by_g  = {g: [] for g in g_list} 

    slot_idx = 0
    for a_i, alpha in enumerate(alpha_list):
        for e_i, eps in enumerate(eps_list):
            for j, g in enumerate(g_list):
                key = keys_sorted[j]
                arr = np.asarray(all_array[key]) 
                y_true = g
                raw = arr[eps_idx[eps], alpha_idx[alpha], :]

                m = np.isfinite(raw) & (raw > 0.0) & (raw < 1.0)
                raw = raw[m]

                if raw.size == 0:
                    ae_for_plot = np.array([ymin])
                    y99_preclip = ymin
                else:
                    ae_raw = np.abs(raw - y_true)            
                    y99_preclip = np.quantile(ae_raw, 0.99)  

                    if ae_raw.size > 1 and 0 < ci < 1:
                        qh = np.quantile(ae_raw, q_hi)
                        trimmed = ae_raw[ae_raw <= qh]
                        ae_trim = trimmed if trimmed.size > 0 else ae_raw
                    else:
                        ae_trim = ae_raw

                    ae_for_plot = np.clip(ae_trim, ymin, None)  

                x = slot_centers[slot_idx] + inner_offsets[j]
                data_by_g[g].append(ae_for_plot)
                y99_by_g[g].append(y99_preclip)
                xpos_by_g[g].append(x)
            slot_idx += 1
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    ax.set_ylabel(r"Absolute error of DP Gini estimate ($|\tilde G - G|$)",
                  fontsize=axis_label_fontsize)
    ax.set_yscale('log')
    ax.set_ylim(bottom=ymin, top=1.0)
    ax.axhline(ymin, ls='--', lw=0.8, color='gray', alpha=0.7)
    ax.grid(axis='y', ls=':', alpha=0.35)

    for g in g_list:
        vp = ax.violinplot(data_by_g[g], positions=xpos_by_g[g], widths=width,
                           showmeans=False, showmedians=False, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(g_colors[g])
            body.set_edgecolor('black')
            body.set_alpha(0.55)
            body.set_linewidth(0.8)

    total_slots = len(slot_centers)
    for k in range(total_slots):
        triplet = [(xpos_by_g[g][k], max(y99_by_g[g][k], ymin), g) for g in g_list]


        logs = np.array([np.log10(v) for (_, v, _) in triplet], dtype=float)
        min_log = float(np.min(logs))
        base_log = min_log + base_pad_dex  

   
        target_logs = base_log + rel_scale * (logs - min_log)

        order = np.argsort(target_logs)
        for i in range(1, len(order)):
            prev = order[i-1]; cur = order[i]
            if target_logs[cur] < target_logs[prev] + min_gap_dex:
                target_logs[cur] = target_logs[prev] + min_gap_dex


        y_texts = 10.0 ** target_logs

 
        jitters = [-jitter_x_pt, 0, +jitter_x_pt]
        for rank, idx in enumerate(order):
            x, y_val, g = triplet[idx]
            y_disp = y_texts[idx]
            text_kwargs = dict(
                ha='center', va='bottom',
                fontsize=label_fontsize,
                color=g_colors[g],
                fontweight='bold',
                zorder=5, clip_on=True
            )
            if add_label_bbox:
                text_kwargs["bbox"] = dict(boxstyle='round,pad=0.15',
                                           fc='white', ec='none', alpha=0.85)

            ax.annotate(
                f"{y_val:.2f}",
                xy=(x, y_disp), xycoords='data',
                xytext=(jitters[rank], pixel_bump_pt), textcoords='offset points',
                **text_kwargs
            )
    ax.set_xticks(slot_centers)
    ax.set_xticklabels([rf"$\varepsilon={e}$" for _ in alpha_list for e in eps_list],
                       fontsize=x_tick_labelsize)
    
    ax.tick_params(axis='y', labelsize=y_tick_labelsize)

    alpha_block_centers = []
    for a_i in range(n_alpha):
        start = a_i * (slots_per_alpha * eps_step + alpha_gap)
        end   = start + (slots_per_alpha - 1) * eps_step
        alpha_block_centers.append(0.5 * (start + end))
        if a_i > 0:
            x_sep = a_i * (slots_per_alpha * eps_step + alpha_gap) - alpha_gap / 2.0
            ax.axvline(x_sep - 0.5*eps_step, ls='--', lw=0.8, color='gray', alpha=0.35)

    for a_i, alpha in enumerate(alpha_list):
        ax.text(alpha_block_centers[a_i], -0.16, f"n = {int(alpha):,}",
                transform=ax.get_xaxis_transform(), ha='center', va='top',
                fontsize=group_label_fontsize)

    handles = [Patch(facecolor=g_colors[g], edgecolor='black', label=f"g = {g:.2f}") for g in g_list]
    leg = ax.legend(handles=handles, loc='upper right', ncol=1, frameon=True,
                    prop={'size': legend_fontsize})
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.show()            

def clip_all_arrays(all_array, inplace=True):
    if inplace:
        for k, v in all_array.items():
            np.clip(v, 0.0, 1.0, out=v)
        return all_array
    else:
        return {k: np.clip(v, 0.0, 1.0) for k, v in all_array.items()}
    
def build_samples_cube(g_real, su_2d, eps_1d, *, gamma=2, n_samples=10_000, rng=None):
    """
    su_2d : DataFrame (rows = beta corresponding to eps; columns = U) or a 2D ndarray
    eps_1d: epsilon vector aligned one-to-one with the rows of su_2d (length = su_2d.shape[0])
    Returns: samples (B, C, n_samples), beta_vals, U_vals, eps
    """
    # Extract data and coordinates
    if isinstance(su_2d, pd.DataFrame):
        beta_vals = np.asarray(su_2d.index, dtype=float)
        U_vals    = np.asarray(su_2d.columns, dtype=float)
        S = su_2d.to_numpy(dtype=float)
    else:
        S = np.asarray(su_2d, dtype=float)
        # If no index/columns are provided, use placeholder coordinates
        beta_vals = np.arange(S.shape[0], dtype=float)
        U_vals    = np.arange(S.shape[1], dtype=float)

    eps = np.asarray(eps_1d, dtype=float).ravel()
    B, C = S.shape
    if eps.size != B:
        raise ValueError(f"Length of eps ({eps.size}) must match the number of rows in su_2d ({B}).")

    # Compute samples cell by cell
    samples = np.empty((B, C, n_samples), dtype=float)
    for i in range(B):
        for j in range(C):
            samples[i, j, :] = simulate_noisy_gini(
                gini_true=g_real,
                S_value=S[i, j],
                epsilon=eps[i],      # Important: use the same epsilon for all columns in the same row
                gamma=gamma,
                n_samples=n_samples,
                rng=rng
            )
    return samples, beta_vals, U_vals, eps


# === Save array (with coordinates) ===
def save_samples_npz(path, samples, beta_vals, U_vals, eps):
    np.savez(path, samples=samples, beta=beta_vals, U=U_vals, eps=eps)

if __name__ == "__main__":
    gamma = 2
    eps = [0.5, 1, 2]
    beta = [cal_beta(eps=i) for i in eps]
    seed = [1]

    n_samples = 1000000
    rng = np.random.default_rng(42)

    out_dir = Path(".")   

    targets = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # 0.8
    n = [10000, 100000, 1000000]
    all_arrays = {}
    for t in targets:
        su_df = pd.DataFrame(index=beta, columns=n, dtype=float)
        for i in n:
            npy_path = ROOT / "data" / f"dataset_g{t:.1f}_n{i}.npy"
            # Try to load from npy, otherwise load from csv and sample

            if npy_path.exists():
                s = np.load(npy_path)
            else:
                print(f"Warning:  {npy_path} not found, skipping g={t}, n={i}")
                continue
            X = np.sort(s.astype(np.float64, copy=False))
            g0  = cal_gini(X)
            L   = float(np.nanmin(X))
            # U = np.array([cal_upper_bound(x=X, alpha=a) for a in alpha], dtype=float)
            U = float(np.nanmax(X))

            su_val = np.array([cal_su_fast(X, beta=b, L=L, U=U)  for b in beta],dtype=float)
            # fill su_val to dataframe su_df
            su_df[i] = su_val

        samples, beta_vals, n_vals, eps_aligned = build_samples_cube(
        g_real=g0,
        su_2d=su_df,
        eps_1d=eps,
        gamma=2,
        n_samples=n_samples)

        key = f"samples_laplace_{int(round(t*100)):02d}"  # 02,03,...,08
        all_arrays[key] = samples
        # all_arrays = clip_all_arrays(all_arrays, inplace=True)

    # Plot after all targets have been processed                                                                                                                             
    plot_abs_error_violin_grouped_by_n_then_eps(all_arrays, ci=0.95)