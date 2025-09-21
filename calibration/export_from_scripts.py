import numpy as np, yaml

def save_camera_yaml(K, dist, path="calibration/camera.yaml"):
    data = {
        "K": K.tolist(),
        "dist": dist.flatten().tolist(),  # k1,k2,p1,p2,[k3]
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def save_extrinsics_yaml(R, t, path="calibration/extrinsics_lidar_to_cam.yaml"):
    data = {
        "R": R.tolist(),
        "t": t.flatten().tolist(),  # meters
        "frame": {"from":"lidar","to":"camera"}
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
