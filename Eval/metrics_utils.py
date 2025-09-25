import numpy as np
from scipy import stats

def compute_warning_flicker(warnings, window_size=25):
    """
    Compute warning flicker rate per 1000 frames.
    Args:
        warnings: Boolean array of warning states
        window_size: Number of frames to consider for transition (25 frames = 1s at 25fps)
    """
    transitions = np.sum(np.abs(np.diff(warnings.astype(int))))
    num_frames = len(warnings)
    return (transitions * 1000) / num_frames

def compute_ttc_stability(ttc_series, fps=25.0):
    """
    Compute TTC stability metrics.
    Returns coefficient of variation for valid TTC values.
    """
    valid_ttc = ttc_series[np.isfinite(ttc_series) & (ttc_series > 0) & (ttc_series < 20)]
    if len(valid_ttc) < 5:
        return np.nan
    return float(np.std(valid_ttc) / (np.mean(valid_ttc) + 1e-6))

def paired_wilcoxon_test(baseline, variant, alpha=0.05):
    """
    Perform paired Wilcoxon signed-rank test.
    Returns: (statistic, p_value, is_significant)
    """
    valid = np.isfinite(baseline) & np.isfinite(variant)
    if np.sum(valid) < 5:
        return np.nan, np.nan, False
    stat, p_val = stats.wilcoxon(baseline[valid], variant[valid])
    return stat, p_val, p_val < alpha