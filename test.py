import numpy as np, cv2, yaml, open3d as o3d

K = np.array([[287.56892752,0,185.39068171],
              [0,305.6650124,159.32885679],
              [0,0,1]], float)
dist = np.array([-0.64326779, 1.65163982, -0.01162166, -0.00946431, -4.17312264], float)
R = np.array([[ 0.9921187503397727,  0.10150178739109593, -0.07346953369023054],
              [ 0.08900867616037746, -0.983591556746742 , -0.1569238830283568],
              [-0.08819206762756057,  0.1491477007966704, -0.9848741658479232]])
t = np.array([[0.05457435578571085],[3.1918041101261347],[1.6940937824130318]])

pcd = o3d.io.read_point_cloud("data/lidar/input (Frame 2756).pcd")
pts = np.asarray(pcd.points, float)
Pc = (R @ pts.T + t).T
print("Camera-frame Z stats (m): mean", Pc[:,2].mean(), "min", Pc[:,2].min(), "max", Pc[:,2].max())
print("Pct points behind camera:", (Pc[:,2] <= 0).mean()*100, "%")


import cv2, yaml, numpy as np

img_path = "data/frames/front/07543.png"
with open("calibration/camera.yaml") as f:
    cal = yaml.safe_load(f)

K   = np.array(cal["K"], float)
dist= np.array(cal["dist"], float).reshape(-1)
Hc, Wc = cal["size"]           # from YAML (height, width)
img = cv2.imread(img_path)
H, W  = img.shape[:2]

print("YAML size:", (Hc, Wc), " | image size:", (H, W))
assert (Hc, Wc) == (H, W), "Mismatch! Recalibrate or resize to the same resolution."
