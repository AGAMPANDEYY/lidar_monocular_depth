import numpy as np
from .configs import RMIN, TTCCLASS_THRESH, DRAC_THRESH_MPS2

def ttc_drac_for_track(tr, cls="vehicle"):
    # ego forward = +Z or +X? (choose one consistently)
    # using camera forward = +Z here; adjust if your convention differs
    r = tr["z"]                        # longitudinal distance
    v = -tr["vz"]                      # approaching speed (>0 if closing)
    if v > 0 and r > 0:
        TTC = r / v
    else:
        TTC = np.inf
    DRAC = (v*v) / (2.0*max(r - RMIN, 1e-3))
    return TTC, DRAC

def ecw_decision(ttc, drac, cls="vehicle"):
    tau = TTCCLASS_THRESH.get(cls, 1.5)
    a_th = DRAC_THRESH_MPS2.get(cls, 3.5)
    risk = 0.0
    # simple logistic blend
    def sig(x): return 1.0/(1.0+np.exp(-x))
    risk += 0.5*sig((tau-ttc)) if np.isfinite(ttc) else 0.0
    risk += 0.5*sig((drac-a_th))
    ECW = (ttc < tau) or (drac > a_th)
    return ECW, float(risk)
