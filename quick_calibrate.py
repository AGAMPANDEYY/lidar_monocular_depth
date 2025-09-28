#!/usr/bin/env python3
import argparse, os, math, itertools, subprocess, yaml, tempfile
from pathlib import Path

def parse_range(s):
    # "a:b:c" inclusive range with float step; supports single value "x"
    if ":" not in s:
        v = float(s)
        return [v]
    a, b, c = s.split(":")
    a, b, c = float(a), float(b), float(c)
    out, x = [], a
    # include b (with small epsilon)
    while (c > 0 and x <= b + 1e-9) or (c < 0 and x >= b - 1e-9):
        out.append(round(x, 10))
        x += c
    return out

def r_x(deg):
    r = math.radians(deg)
    return [[1,0,0],[0,math.cos(r),-math.sin(r)],[0,math.sin(r),math.cos(r)]]

def r_y(deg):
    r = math.radians(deg)
    return [[math.cos(r),0,math.sin(r)],[0,1,0],[-math.sin(r),0,math.cos(r)]]

def r_z(deg):
    r = math.radians(deg)
    return [[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]]

def matmul(A,B):
    return [[sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def matadd(a,b):  # 3-vectors
    return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]

def load_extrinsics(yaml_path):
    with open(yaml_path,"r") as f:
        ext = yaml.safe_load(f)
    R = ext["R"]
    t = ext["t"] if isinstance(ext["t"], list) else ext["t"][0]
    # normalize shapes
    R = [[float(x) for x in row] for row in R]
    t = [float(x) for x in t]
    return R,t,ext

def save_extrinsics(R,t,template,save_path):
    obj = dict(template)
    obj["R"] = R
    obj["t"] = t
    with open(save_path,"w") as f:
        yaml.safe_dump(obj, f)

def run_projection(tool, pcd, image, cam_yaml, ext_yaml, out_dir, cx=None, cy=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image).stem
    overlay = out_dir / f"{stem}_overlay.png"
    npz = out_dir / f"{stem}.npz"
    cmd = [
        "python", tool,
        "--pcd", pcd,
        "--cam_yaml", cam_yaml,
        "--ext_yaml", str(ext_yaml),
        "--image", image,
        "--out_npz", str(npz),
        "--debug_overlay", str(overlay),
    ]
    if cx is not None:
        cmd += ["--manual_cx_offset", str(int(cx))]
    if cy is not None:
        cmd += ["--manual_cy_offset", str(int(cy))]
    print("  ->", " ".join(cmd))
    subprocess.run(cmd, check=False)
    return overlay, npz

def main():
    ap = argparse.ArgumentParser(description="Grid/refine sweep for LiDAR->Camera extrinsics (write temp YAMLs and call project_lidar.py)")
    ap.add_argument("--tool", default="lidar_projection/project_lidar.py", help="projection tool to call")
    ap.add_argument("--pcd", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--cam_yaml", required=True)
    ap.add_argument("--ext_yaml", required=True, help="base extrinsics YAML (R,t)")
    ap.add_argument("--yaw", default="0", help="deg range e.g. -6:6:2 (about camera Y)")
    ap.add_argument("--pitch", default="0", help="deg range e.g. -6:6:2 (about camera X)")
    ap.add_argument("--roll", default="0", help="deg range e.g. -3:3:1 (about camera Z)")
    ap.add_argument("--tx", default="0", help="m range e.g. -0.10:0.10:0.02")
    ap.add_argument("--ty", default="0", help="m range")
    ap.add_argument("--tz", default="0", help="m range")
    ap.add_argument("--cx", default="0", help="px range e.g. -20:20:5")
    ap.add_argument("--cy", default="0", help="px range")
    ap.add_argument("--out_dir", default="calib_sweep_out")
    ap.add_argument("--refine", type=float, default=0.0, help="optional +/- delta around best yaw/pitch/roll (deg) & t (m)")
    args = ap.parse_args()

    base_R, base_t, tmpl = load_extrinsics(args.ext_yaml)

    # Build grids
    Y = parse_range(args.yaw)
    P = parse_range(args.pitch)
    Rr = parse_range(args.roll)
    TX = parse_range(args.tx)
    TY = parse_range(args.ty)
    TZ = parse_range(args.tz)
    CX = parse_range(args.cx)
    CY = parse_range(args.cy)

    out_root = Path(args.out_dir); out_root.mkdir(exist_ok=True)
    print(f"[INFO] sweep sizes yaw/pitch/roll={len(Y)}/{len(P)}/{len(Rr)} ; t={len(TX)}x{len(TY)}x{len(TZ)} ; cxy={len(CX)}x{len(CY)}")

    best_combo = None
    best_count = -1  # simple score: prefer more projected points inside image if your tool records that in NPZ; else it's manual

    idx = 0
    for yaw, pitch, roll in itertools.product(Y,P,Rr):
        # delta rotation in camera axes: R_delta = Rz(roll) * Rx(pitch) * Ry(yaw)
        R_delta = matmul(r_z(roll), matmul(r_x(pitch), r_y(yaw)))
        R_try = matmul(R_delta, base_R)
        for tx,ty,tz in itertools.product(TX,TY,TZ):
            t_try = matadd(base_t, [tx,ty,tz])
            # write tmp YAML
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
                save_extrinsics(R_try, t_try, tmpl, tf.name)
                tmp_yaml = tf.name
            for cx, cy in itertools.product(CX,CY):
                idx += 1
                tag = f"y{yaw}_p{pitch}_r{roll}_tx{tx}_ty{ty}_tz{tz}_cx{cx}_cy{cy}".replace("-","m").replace(".","p")
                out_dir = out_root / tag
                print(f"\n[{idx}] {tag}")
                overlay, npz = run_projection(args.tool, args.pcd, args.image, args.cam_yaml, tmp_yaml, out_dir, cx, cy)
                # NOTE: scoring is left manual unless your NPZ contains a 'n_projected' or mask; inspect overlays and pick the best

    if args.refine > 0:
        print("\n[INFO] Refinement pass requested. Inspect overlays in", out_root, "pick best tag, then rerun with narrower ranges around it.")

if __name__ == "__main__":
    main()
