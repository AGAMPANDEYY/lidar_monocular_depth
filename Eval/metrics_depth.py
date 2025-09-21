def rmse_rel(pred, gt, mask):
    p=pred[mask]; g=gt[mask]
    import numpy as np
    rmse = np.sqrt(np.mean((p-g)**2))
    rel = np.mean(np.abs(p-g)/np.maximum(g,1e-3))
    return rmse, rel
