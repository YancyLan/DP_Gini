from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("DPGini")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

from .utils import sort_X, cal_gini as gini, cal_gini
from .core import (
    cal_beta,
    smooth_upper_bound,
    cal_su,
    cal_su_fast,
    fast_min_gini,
    fast_max_gini,
    add_dp_noise_gini,
    simulate_noisy_gini,
    sample_eta,
    sample_exponential_noise,
    unbounded_quantile_mech,
    above_threshold,
    laplace,
)

__all__ = [
    "gini", "cal_gini", "sort_X",
    "cal_beta", "smooth_upper_bound", "cal_su", "cal_su_fast",
    "fast_min_gini", "fast_max_gini",
    "add_dp_noise_gini", "simulate_noisy_gini",
    "sample_eta", "sample_exponential_noise",
    "unbounded_quantile_mech", "above_threshold", "laplace",
    "__version__",
]