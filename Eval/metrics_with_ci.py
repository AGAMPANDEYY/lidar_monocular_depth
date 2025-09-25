import numpy as np

EPS = 1e-6

def compute_metrics_with_ci(pred, gt, n_bootstrap=1000):
    """
    Compute depth metrics with confidence intervals using bootstrapping.
    Returns mean values and 95% confidence intervals for abs_rel and rmse.
    """
    if len(pred) < 100:
        return {
            'abs_rel_mean': np.nan,
            'rmse_mean': np.nan,
            'abs_rel_ci': (np.nan, np.nan),
            'rmse_ci': (np.nan, np.nan)
        }
        
    # Initialize bootstrap arrays
    bootstrap_abs_rel = np.zeros(n_bootstrap)
    bootstrap_rmse = np.zeros(n_bootstrap)
    n_samples = len(pred)
    
    # Perform bootstrap iterations
    for i in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, n_samples, replace=True)
        p_sample = pred[indices]
        g_sample = gt[indices]
        
        # Compute metrics for this sample
        abs_rel = np.mean(np.abs(p_sample - g_sample) / np.maximum(g_sample, EPS))
        rmse = np.sqrt(np.mean((p_sample - g_sample) ** 2))
        
        bootstrap_abs_rel[i] = abs_rel
        bootstrap_rmse[i] = rmse
    
    # Compute means and confidence intervals
    abs_rel_mean = np.mean(bootstrap_abs_rel)
    rmse_mean = np.mean(bootstrap_rmse)
    
    abs_rel_ci = np.percentile(bootstrap_abs_rel, [2.5, 97.5])
    rmse_ci = np.percentile(bootstrap_rmse, [2.5, 97.5])
    
    return {
        'abs_rel_mean': abs_rel_mean,
        'rmse_mean': rmse_mean,
        'abs_rel_ci': tuple(abs_rel_ci),
        'rmse_ci': tuple(rmse_ci)
    }