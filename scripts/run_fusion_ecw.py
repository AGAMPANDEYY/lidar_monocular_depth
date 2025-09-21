
import os, json, glob, numpy as np, cv2
from fusion_ecw.fuser import OnlineFuser
from fusion_ecw.track3d import Track3DEstimator
from fusion_ecw.ttc_drac import ttc_drac_for_track, ecw_decision
from lidar_projection.project_lidar import load_calib, project_points_xyz_to_depth_img

# Load calibration
K, dist, R, t = load_calib()
fuser = OnlineFuser()
t3d = Track3DEstimator()

frames = sorted(glob.glob("data/frames/*.png"))
for i, f in enumerate(frames):
    stem = os.path.splitext(os.path.basename(f))[0]
    I = cv2.imread(f); H,W = I.shape[:2]

    # detections
    dets = json.load(open(f"data/dets/{stem}.dets.json"))

    # monocular depth
    Dmono = np.load(f"data/mono/{stem}.mono.npz")["depth"].astype(np.float32)

    # lidar: either projected depth/mask or raw xyz
    lid_path = f"data/lidar/{stem}.lidar.npz"
    L = np.load(lid_path)
    if "depth" in L and "mask" in L:
        Dlidar = L["depth"].astype(np.float32)
        Mlidar = L["mask"].astype(bool)
    else:
        pts_xyz = L["xyz"].astype(np.float32)   # Nx3 in LiDAR frame
        Dlidar, Mlidar = project_points_xyz_to_depth_img(pts_xyz, K, R, t, (H,W))

    # fuse
    Dfused, stats = fuser.fuse_frame(Dmono, Dlidar, Mlidar, dets)

    # tracks → TTC/DRAC
    tstamp = i/15.0  # or real timestamps
    tracks3d = t3d.update(dets, Dfused, K, tstamp)
    ecw_any = False
    for tr in tracks3d:
        ttc, drac = ttc_drac_for_track(tr, cls="vehicle")  # map class if you have it
        ECW, risk = ecw_decision(ttc, drac, cls="vehicle")
        tr.update({"ttc":ttc, "drac":drac, "ecw":ECW, "risk":risk})
        ecw_any |= ECW

    # save outputs for metrics & figs
    out = {"stats":stats, "tracks3d":tracks3d, "ecw":bool(ecw_any)}
    os.makedirs("data/out", exist_ok=True)
    json.dump(out, open(f"data/out/{stem}.fusion.json","w"))

    # quick overlay (optional): draw depth colormap, boxes, TTC text, ECW banner
    # (implement in fusion_ecw/viz.py to keep code tidy)

    print(f"{stem} ECW={ecw_any} tracks={len(tracks3d)}")
