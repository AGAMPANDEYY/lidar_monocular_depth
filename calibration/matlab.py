import scipy.io as sio
import numpy as np
import yaml

# Load the plain .mat file
mat = sio.loadmat('calibration/calib_export.mat')  # adjust path

R = mat['R']             # (3x3)
t = mat['t'].reshape(3)  # (3,)
K = mat['K']             # (3x3)
dist = mat['dist'].reshape(-1)  # (5,) [k1 k2 p1 p2 k3]
image_size = mat['imageSize'].reshape(-1)  # [H W]

print("K:\n", K)
print("R:\n", R)
print("t:\n", t)
print("dist:", dist)
