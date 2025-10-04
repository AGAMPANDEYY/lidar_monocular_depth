#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, subprocess, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- Utility ---------------------------------

def slugify(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '-', name.lower()).strip('-')

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def run_cmd(cmd: list, env=None):
    print("\n[RUN]", " ".join(str(c) for c in cmd))
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed with code {ret.returncode}")

def list_debug_depths(debug_dir: Path):
    """Return aligned lists of (frame_id, lidar, mono, fused) numpy files where available."""
    lidar_files = sorted(debug_dir.glob("*_lidar_depth.npy"))
    rows = []
    for lf in lidar_files:
        frame = lf.name.split("_")[0]
        fused = debug_dir / f"{frame}_fused_depth.npy"
        mono  = debug_dir / f"{frame}_mono_depth.npy"
        rows.append((frame, lf, mono if mono.exists() else None, fused if fused.exists() else None))
    return rows

def absrel_rmse_by_range(lidar_depth, pred_depth):
    """Compute AbsRel & RMSE overall and in bins:
       0–10, 10–25, 25–50, >50 (metres), LiDAR-valid pixels only."""
    mask = np.isfinite(lidar_depth) & (lidar_depth > 0) & np.isfinite(pred_depth)
    if not mask.any():
        return {}
    z = lidar_depth[mask]
    p = pred_depth[mask]
    def metrics(sel):
        if sel.sum() == 0:
            return (np.nan, np.nan, 0)
        zz, pp = z[sel], p[sel]
        absrel = float(np.mean(np.abs(zz-pp)/zz))
        rmse   = float(np.sqrt(np.mean((zz-pp)**2)))
        return (absrel, rmse, int(sel.sum()))
    out = {}
    sel_all = np.ones_like(z, dtype=bool)
    out["all"]   = metrics(sel_all)
    out["0_10"]  = metrics((z>=0)   & (z<10))
    out["10_25"] = metrics((z>=10)  & (z<25))
    out["25_50"] = metrics((z>=25)  & (z<50))
    out["50p"]   = metrics((z>=50))
    return out

def seq_from_frame(frame_str, bucket=1000):
    try:
        f = int(frame_str)
        return f"seq_{(f//bucket)*bucket:05d}"
    except:
        return "seq_all"

def bootstrap_ci(values, groups, n_boot=5000, alpha=0.05):
    """Sequence-level bootstrap CI for a vector of per-pixel errors with sequence ids."""
    df = pd.DataFrame({"g": groups, "v": values})
    per_seq = df.groupby("g")["v"].mean().dropna().values
    if per_seq.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(per_seq))
    idx = np.random.randint(0, len(per_seq), size=(n_boot, len(per_seq)))
    boots = np.mean(per_seq[idx], axis=1)
    lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
    return (mean, float(lo), float(hi))

def runtime_stats(timing_csv: Path):
    if not timing_csv.exists():
        return {}
    d = pd.read_csv(timing_csv)
    out = {
        "Det_mean_ms": float(np.mean(d["t_det_ms"])) if "t_det_ms" in d else np.nan,
        "Depth_mean_ms": float(np.mean(d["t_depth_ms"])) if "t_depth_ms" in d else np.nan,
        "Fusion_mean_ms": float(np.mean(d["t_fuse_ms"])) if "t_fuse_ms" in d else np.nan,
        "ECW_mean_ms": float(np.mean(d["t_ecw_ms"])) if "t_ecw_ms" in d else np.nan,
        "Total_mean_ms": float(np.mean(d["t_total_ms"])) if "t_total_ms" in d else np.nan,
        "Total_95p_ms": float(np.percentile(d["t_total_ms"], 95)) if "t_total_ms" in d else np.nan,
        "FPS": float(1000.0/np.mean(d["t_total_ms"])) if "t_total_ms" in d and np.mean(d["t_total_ms"])>0 else np.nan
    }
    return out

# ----------------------------- Warning/box metrics ----------------------

def load_obj_df(out_dir: Path):
    p = out_dir / "object_depth_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    # Ensure numeric
    for c in ["mono_median_depth","lidar_median_depth","fused_median_depth","ttc","ego_speed","confidence"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Standardize frame order
    if "frame" in df.columns:
        df["frame_idx"] = pd.to_numeric(df["frame"], errors="coerce")
    return df

def box_mae_by_range(method_name: str, df: pd.DataFrame):
    """Box-level MAE/RMSE using per-box medians; bins by LiDAR depth (gt)."""
    if df is None or df.empty: 
        return pd.DataFrame()
    d = df.copy()
    d["gt"] = d["lidar_median_depth"]
    d["pred"] = d["fused_median_depth"]  # in each run this is the method's depth
    m = d["gt"].notna() & d["pred"].notna() & (d["gt"]>0) & (d["pred"]>0)
    d = d[m].copy()
    if d.empty: 
        return pd.DataFrame()
    bins = pd.cut(d["gt"], bins=[0,10,25,50,1e9], labels=["0_10","10_25","25_50","50p"], right=False)
    d["Bin"] = bins
    d["AE"] = np.abs(d["pred"] - d["gt"])
    d["SE"] = (d["pred"] - d["gt"])**2
    out = d.groupby("Bin").agg(
        MAE=("AE","mean"),
        RMSE=("SE", lambda x: float(np.sqrt(np.mean(x)))),
        Count=("AE","size")
    ).reset_index()
    out.insert(0, "Method", method_name)
    return out

def warning_metrics(method_name: str, df: pd.DataFrame):
    """Per-frame micro ECW metrics using warn_stable_fused vs in_ecw; plus flicker per 1k frames."""
    if df is None or df.empty: 
        return dict(Method=method_name, Precision=np.nan, Recall=np.nan, F1=np.nan, Flicker_per1k=np.nan)
    if "warn_stable_fused" not in df or "in_ecw" not in df:
        return dict(Method=method_name, Precision=np.nan, Recall=np.nan, F1=np.nan, Flicker_per1k=np.nan)

    w = df["warn_stable_fused"].astype(str).str.lower().isin(["true","1","yes"]) if df["warn_stable_fused"].dtype==object else df["warn_stable_fused"].astype(bool)
    g = df["in_ecw"].astype(str).str.lower().isin(["true","1","yes"]) if df["in_ecw"].dtype==object else df["in_ecw"].astype(bool)

    tp = int(np.sum(w & g))
    fp = int(np.sum(w & ~g))
    fn = int(np.sum(~w & g))
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

    # Flicker per 1k frames (macro average across tracks)
    flick_rates = []
    if "obj_id" in df and "frame_idx" in df:
        for oid, gdf in df.groupby("obj_id"):
            gdf = gdf.sort_values("frame_idx")
            s = w.loc[gdf.index].astype(int).values
            if s.size >= 2:
                flips = int(np.sum(np.diff(s)!=0))
                rate = 1000.0 * flips / float(s.size)
                flick_rates.append(rate)
    flick = float(np.mean(flick_rates)) if flick_rates else np.nan

    return dict(Method=method_name, Precision=prec, Recall=rec, F1=f1, Flicker_per1k=flick)

def ttc_cv_distribution(df: pd.DataFrame):
    """Return list of CV(TTC) per track for violin plots."""
    if df is None or df.empty or "ttc" not in df or "obj_id" not in df:
        return []
    cv_list = []
    for oid, g in df.groupby("obj_id"):
        t = pd.to_numeric(g["ttc"], errors="coerce").dropna().values
        t = t[np.isfinite(t)]
        if t.size >= 5:
            mu = np.mean(t)
            if mu != 0:
                cv = float(np.std(t)/abs(mu))
                if np.isfinite(cv):
                    cv_list.append(cv)
    return cv_list

# ----------------------------- Methods matrix ---------------------------

METHODS = [
    # name                              fusion_mode   ema  mining  sanity  ecw_source
    ("Monocular-Only (Scaled)",         "mono",       False, True,  True,  "mono"),
    ("LiDAR-Only (Projected)",          "lidar",      False, True,  True,  "lidar"),
    ("Late Fusion (No Conf/EMA)",       "late",       False, True,  True,  "fused"),
    ("Ours (Conf + EMA)",               "ours",       True,  True,  True,  "fused"),
    ("Ablation: No EMA",                "ours",       False, True,  True,  "fused"),
    ("Ablation: No Mining",             "ours",       True,  False, True,  "fused"),
    ("Ablation: No Sanity Checks",      "ours",       True,  True,  False, "fused"),
]

# ----------------------------- Runner -----------------------------------

def run_one(main_py: Path,
            out_root: Path,
            frames_cfg: dict,
            base_args: dict,
            name: str,
            fusion_mode: str,
            ema: bool,
            mining: bool,
            sanity: bool,
            ecw_source: str,
            hysteresis: float):

    tag = slugify(name)
    out_dir = out_root / tag
    ensure_dir(out_dir)

    # Build command for main.py (requires small flags added; see section 2 below)
    cmd = [
        sys.executable, str(main_py),
        "--camera_start", str(frames_cfg["camera_start"]),
        "--camera_end",   str(frames_cfg["camera_end"]),
        "--lidar_start",  str(frames_cfg["lidar_start"]),
        "--lidar_end",    str(frames_cfg["lidar_end"]),
        "--camera_fps",   str(base_args["fps"]),
        "--lidar_fps",    str(base_args["lidar_fps"]),
        "--max_frames",   str(base_args["max_frames"]),
        "--depth_backend", base_args["depth_backend"],
        "--out_dir",      str(out_dir),
        "--fusion_mode",  fusion_mode,
        "--ecw_source",   ecw_source,
        "--hysteresis",   str(hysteresis),
    ]
    if not ema:     cmd += ["--no_ema"]
    if not mining:  cmd += ["--no_mining"]
    if not sanity:  cmd += ["--no_sanity"]

    # Optionally use fused map for ECW if ecw_source == fused (main.py will honor ecw_source)
    run_cmd(cmd)

    # Collect metrics
    dbg_dir = out_dir / "debug"
    obj_csv = out_dir / "object_depth_metrics.csv"
    tim_csv = out_dir / "timing.csv"

    # Depth metrics from debug arrays
    rows = list_debug_depths(dbg_dir)
    per_range_rows = []
    absrel_values, absrel_groups = [], []
    for frame, lidar_path, mono_path, fused_path in rows:
        L = np.load(lidar_path)      # (H,W) meters; zeros where invalid
        if fusion_mode == "mono":
            if mono_path is None: 
                continue
            P = np.load(mono_path)
        elif fusion_mode == "lidar":
            P = L.copy()
        elif fusion_mode == "late":
            if mono_path is None: 
                continue
            M = np.load(mono_path)
            P = 0.5*(M + L)  # simple average
        else:  # ours
            if fused_path is None: 
                continue
            P = np.load(fused_path)

        # compute stratified metrics for this frame
        stats = absrel_rmse_by_range(L, P)
        if not stats: 
            continue
        # flatten per-pixel absrel for seq bootstrap
        mask = np.isfinite(L) & (L>0) & np.isfinite(P)
        if mask.any():
            absrel = np.abs(L[mask]-P[mask])/L[mask]
            absrel_values.append(absrel)
            absrel_groups.append(np.full(absrel.size, seq_from_frame(frame)))
        # pack per-range
        for bin_name, (ar, rm, n) in stats.items():
            per_range_rows.append({
                "Method": name,
                "Frame": frame,
                "Bin": bin_name,
                "AbsRel": ar,
                "RMSE": rm,
                "N": n
            })

    if len(absrel_values):
        absrel_values = np.concatenate(absrel_values)
        absrel_groups = np.concatenate(absrel_groups)
        mean_absrel, lo, hi = bootstrap_ci(absrel_values, absrel_groups, n_boot=3000)
    else:
        mean_absrel, lo, hi = (np.nan, np.nan, np.nan)

    rt = runtime_stats(tim_csv)

    # Object-level derived metrics (ECW, flicker, box MAE, TTC CV dist)
    obj_df = load_obj_df(out_dir)
    warn = warning_metrics(name, obj_df)
    box  = box_mae_by_range(name, obj_df)
    ttc_cv_dist = ttc_cv_distribution(obj_df)

    summary = dict(
        Method=name, AbsRel_mean=mean_absrel, AbsRel_CI_lo=lo, AbsRel_CI_hi=hi,
        TTC_CV_mean=float(np.mean(ttc_cv_dist)) if ttc_cv_dist else np.nan,
        **rt, OutDir=str(out_dir),
        Precision=warn["Precision"], Recall=warn["Recall"], F1=warn["F1"], Flicker_per1k=warn["Flicker_per1k"]
    )
    return per_range_rows, summary, box, ttc_cv_dist

# ----------------------------- Plots ------------------------------------

def plot_absrel_by_range(df, out_png):
    # df columns: Method, Bin, AbsRel (per-frame). We'll aggregate mean over frames per method/bin.
    agg = df.groupby(["Method","Bin"])["AbsRel"].mean().reset_index()
    bins = ["0_10","10_25","25_50","50p"]
    methods = list(agg["Method"].unique())

    plt.figure(figsize=(8,5))
    width = 0.16
    x = np.arange(len(bins))
    for i, m in enumerate(methods):
        y = [(agg[(agg.Method==m)&(agg.Bin==b)]["AbsRel"].values[0] if len(agg[(agg.Method==m)&(agg.Bin==b)])>0 else np.nan) for b in bins]
        plt.bar(x + (i - len(methods)/2)*width + width/2, y, width=width, label=m)
    plt.xticks(x, ["0–10","10–25","25–50",">50"])
    plt.xlabel("Range (m)")
    plt.ylabel("AbsRel (lower is better)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

def plot_box_mae_by_range(box_df, out_png):
    bins = ["0_10","10_25","25_50","50p"]
    methods = list(box_df["Method"].unique())
    plt.figure(figsize=(8,5))
    width = 0.16
    x = np.arange(len(bins))
    for i, m in enumerate(methods):
        sub = box_df[box_df["Method"]==m]
        y = [float(sub[sub["Bin"]==b]["MAE"].mean()) if not sub[sub["Bin"]==b].empty else np.nan for b in bins]
        plt.bar(x + (i - len(methods)/2)*width + width/2, y, width=width, label=m)
    plt.xticks(x, ["0–10","10–25","25–50",">50"])
    plt.xlabel("Range (m)")
    plt.ylabel("Box-level MAE (m) ↓")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

def plot_ttc_cv_violin(cv_dict, out_png):
    methods = list(cv_dict.keys())
    data = [cv_dict[m] if cv_dict[m] else [np.nan] for m in methods]
    plt.figure(figsize=(8,5))
    parts = plt.violinplot(data, showmeans=True, showextrema=False)
    plt.xticks(np.arange(1, len(methods)+1), methods, rotation=15)
    plt.ylabel("CV(TTC) ↓")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

def plot_runtime_stacked(summary_df, out_png):
    comp = ["Det_mean_ms","Depth_mean_ms","Fusion_mean_ms","ECW_mean_ms"]
    methods = summary_df["Method"].tolist()
    vals = summary_df[comp].fillna(0.0).values
    plt.figure(figsize=(8,5))
    bottom = np.zeros(len(methods))
    for i, c in enumerate(comp):
        plt.bar(methods, vals[:,i], bottom=bottom, label=c.replace("_mean_ms","").replace("_"," ").title())
        bottom += vals[:,i]
    plt.ylabel("Latency (ms) ↓")
    plt.xticks(rotation=10)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

def plot_flicker_bar(warn_table, out_png):
    plt.figure(figsize=(6,4))
    plt.bar(warn_table["Method"], warn_table["Flicker_per1k"])
    plt.ylabel("Warning Flicker / 1k frames ↓")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

def plot_f1_bar(warn_table, out_png):
    plt.figure(figsize=(6,4))
    plt.bar(warn_table["Method"], warn_table["F1"])
    plt.ylabel("ECW F1 ↑")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

# ----------------------------- Main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run main.py across baselines/ablations and aggregate metrics.")
    ap.add_argument("--main_py", default="main.py")
    ap.add_argument("--out_root", default="experiments")
    ap.add_argument("--paper_out", default="paper_out")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--lidar_fps", type=float, default=10.0)
    ap.add_argument("--depth_backend", choices=["midas","zoe","fastdepth","depth-anything-v2","monodepth2"], default="midas")
    ap.add_argument("--camera_start", type=int, default=15000)
    ap.add_argument("--camera_end",   type=int, default=16500)
    ap.add_argument("--lidar_start",  type=int, default=6000)
    ap.add_argument("--lidar_end",    type=int, default=6600)
    ap.add_argument("--hysteresis", type=float, default=0.5)
    ap.add_argument("--only", nargs="*", default=None, help="Subset of method names to run (exact match).")
    args = ap.parse_args()

    main_py   = Path(args.main_py)
    out_root  = Path(args.out_root)
    paper_out = Path(args.paper_out)
    ensure_dir(out_root); ensure_dir(paper_out)

    frames_cfg = dict(
        camera_start=args.camera_start, camera_end=args.camera_end,
        lidar_start=args.lidar_start,   lidar_end=args.lidar_end
    )
    base_args = dict(
        max_frames=args.max_frames, fps=args.fps,
        lidar_fps=args.lidar_fps, depth_backend=args.depth_backend
    )

    all_rows = []
    summary_rows  = []
    box_rows = []
    ttc_cv_map = {}  # method -> list of cv values
    to_run = [m for m in METHODS if args.only is None or m[0] in args.only]

    print("\n=== Running methods ===")
    for name, fusion_mode, ema, mining, sanity, ecw_source in to_run:
        per_rows, summary_row, box_df, cv_list = run_one(
            main_py, out_root, frames_cfg, base_args,
            name, fusion_mode, ema, mining, sanity, ecw_source, args.hysteresis
        )
        all_rows.extend(per_rows)
        summary_rows.append(summary_row)
        if box_df is not None and not box_df.empty:
            box_rows.append(box_df)
        ttc_cv_map[name] = cv_list

    # ---------------- Save tables ----------------
    depth_df = pd.DataFrame(all_rows)
    depth_df.to_csv(paper_out/"depth_by_range_perframe.csv", index=False)

    # Table 1 aggregate (mean over frames), stratified by range
    t1 = depth_df.groupby(["Method","Bin"]).agg(
        AbsRel_mean=("AbsRel","mean"),
        RMSE_mean=("RMSE","mean"),
        N_sum=("N","sum")
    ).reset_index()
    t1.to_csv(paper_out/"table1_depth_by_range.csv", index=False)

    # Summary table (AbsRel mean±CI, TTC CV, runtime)
    tsummary = pd.DataFrame(summary_rows)
    tsummary.to_csv(paper_out/"table_summary_overall.csv", index=False)

    # Runtime table
    t4 = tsummary[["Method","Det_mean_ms","Depth_mean_ms","Fusion_mean_ms","ECW_mean_ms","Total_mean_ms","Total_95p_ms","FPS"]].copy()
    t4.to_csv(paper_out/"table4_runtime.csv", index=False)

    if box_rows:
        tbox = pd.concat(box_rows, ignore_index=True)
        tbox.to_csv(paper_out/"table_box_mae_by_range.csv", index=False)
    else:
        tbox = pd.DataFrame(columns=["Method","Bin","MAE","RMSE","Count"])

    # ECW metrics table (Precision/Recall/F1/Flicker taken from summary_rows)
    twarn = tsummary[["Method","Precision","Recall","F1","Flicker_per1k"]].copy()
    twarn.to_csv(paper_out/"table2_ecw_metrics.csv", index=False)

    # Improvement vs Late Fusion (No Conf/EMA)
    try:
        late = tsummary.set_index("Method").loc["Late Fusion (No Conf/EMA)"]
        ours = tsummary.set_index("Method").loc["Ours (Conf + EMA)"]
        imp = {
            "AbsRel_mean_%improve": 100.0 * (late["AbsRel_mean"] - ours["AbsRel_mean"]) / late["AbsRel_mean"] if pd.notna(late["AbsRel_mean"]) and pd.notna(ours["AbsRel_mean"]) else np.nan,
            "TTC_CV_%improve":      100.0 * (late["TTC_CV_mean"] - ours["TTC_CV_mean"]) / late["TTC_CV_mean"] if pd.notna(late["TTC_CV_mean"]) and pd.notna(ours["TTC_CV_mean"]) else np.nan,
            "F1_%improve":          100.0 * (ours["F1"] - late["F1"]) / (late["F1"]+1e-9) if pd.notna(late["F1"]) and pd.notna(ours["F1"]) else np.nan,
            "FPS_%improve":         100.0 * (ours["FPS"] - late["FPS"]) / (late["FPS"]+1e-9) if pd.notna(late["FPS"]) and pd.notna(ours["FPS"]) else np.nan,
        }
        pd.DataFrame([imp]).to_csv(paper_out/"table_improvement_vs_late.csv", index=False)
    except Exception as e:
        print("[WARN] Could not compute improvement vs Late Fusion:", e)

    # ---------------- Figures -------------------
    plot_absrel_by_range(depth_df, paper_out/"fig_absrel_by_range.png")
    if not tbox.empty:
        plot_box_mae_by_range(tbox, paper_out/"fig_box_mae_by_range.png")
    plot_runtime_stacked(tsummary, paper_out/"fig_runtime_stacked.png")
    plot_flicker_bar(twarn, paper_out/"fig_warning_flicker.png")
    plot_f1_bar(twarn, paper_out/"fig_ecw_f1.png")
    plot_ttc_cv_violin(ttc_cv_map, paper_out/"fig_ttc_cv_violin.png")

    # --------------- Done -----------------------
    print("\nDone.")
    print("Tables:")
    for f in [
        "depth_by_range_perframe.csv","table1_depth_by_range.csv",
        "table_summary_overall.csv","table4_runtime.csv",
        "table_box_mae_by_range.csv","table2_ecw_metrics.csv","table_improvement_vs_late.csv"
    ]:
        print("  -", paper_out/f)
    print("Figures:")
    for f in [
        "fig_absrel_by_range.png","fig_box_mae_by_range.png","fig_runtime_stacked.png",
        "fig_warning_flicker.png","fig_ecw_f1.png","fig_ttc_cv_violin.png"
    ]:
        print("  -", paper_out/f)

if __name__ == "__main__":
    main()
