import numpy as np

def projection_qc(z_front_frac=None, proj_finite=None, in_bounds=None, uv_shape=None, overlap=None):
    """
    Returns (quality_score [0..1], flags dict).
    Any arg can be None; we degrade gracefully.
    """
    H = W = None
    if uv_shape is not None:
        H, W = uv_shape
    flags = {}
    score = 1.0

    if z_front_frac is not None and z_front_frac < 0.6:
        score -= 0.5
        flags["cheirality"] = True

    if in_bounds is not None and H is not None and W is not None:
        cov = in_bounds / float(max(1, H * W))
        if cov < 0.01:
            score -= 0.3
            flags["coverage"] = True

    if overlap is not None and overlap < 100:
        score -= 0.2
        flags["overlap_low"] = True

    return max(0.0, min(1.0, score)), flags
