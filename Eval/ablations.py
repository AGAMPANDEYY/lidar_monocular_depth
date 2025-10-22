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

def _predicted_depth_series(method_name: str, df: pd.DataFrame):
    """Select the appropriate depth prediction column for a given method."""
    lname = method_name.lower()
    if df is None or df.empty:
        return pd.Series(dtype=float)
    if "monocular" in lname:
        return df.get("mono_median_depth", pd.Series(dtype=float))
    if "lidar-only" in lname:
        return df.get("lidar_median_depth", pd.Series(dtype=float))
    if "late fusion" in lname:
        mono = df.get("mono_median_depth")
        lidar = df.get("lidar_median_depth")
        if mono is not None and lidar is not None:
            return 0.5 * (mono + lidar)
    # Default: use explicit prediction column if present, else fused depth
    if "pred_median_depth" in df.columns:
        return df["pred_median_depth"]
    return df.get("fused_median_depth", pd.Series(dtype=float))

def box_mae_by_range(method_name: str, df: pd.DataFrame):
    """Box-level MAE/RMSE using per-box medians; bins by LiDAR depth (gt)."""
    if df is None or df.empty: 
        return pd.DataFrame()
    d = df.copy()
    d["gt"] = d["lidar_median_depth"]
    pred_series = _predicted_depth_series(method_name, d)
    d["pred"] = pred_series
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
    warn_col = "warn_stable_fused" if "warn_stable_fused" in df else ("warn_stable" if "warn_stable" in df else None)
    raw_col = "warn_raw_fused" if "warn_raw_fused" in df else ("warn_raw" if "warn_raw" in df else None)
    if warn_col is None or "in_ecw" not in df:
        return dict(Method=method_name, Precision=np.nan, Recall=np.nan, F1=np.nan, Flicker_per1k=np.nan)

    warn_series = df[warn_col]
    w = warn_series.astype(str).str.lower().isin(["true","1","yes"]) if warn_series.dtype==object else warn_series.astype(bool)
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

def stratified_ttc_cv(method_name: str, df: pd.DataFrame):
    bins = {
        "overall": [],
        "0_25": [],
        "25_50": [],
        "50p": []
    }
    if df is None or df.empty or "ttc" not in df or "obj_id" not in df:
        return bins
    d = df.copy()
    d["ttc"] = pd.to_numeric(d["ttc"], errors="coerce")
    pred = _predicted_depth_series(method_name, d)
    d["pred_depth"] = pd.to_numeric(pred, errors="coerce")
    for oid, g in d.groupby("obj_id"):
        t = g["ttc"].dropna().values
        t = t[np.isfinite(t)]
        if t.size < 5:
            continue
        mu = np.mean(t)
        if not np.isfinite(mu) or mu == 0:
            continue
        cv = float(np.std(t)/abs(mu))
        if not np.isfinite(cv):
            continue
        bins["overall"].append(cv)
        depths = g["pred_depth"].dropna().values
        if depths.size == 0:
            continue
        dmed = float(np.median(depths))
        if dmed < 25:
            bins["0_25"].append(cv)
        elif dmed < 50:
            bins["25_50"].append(cv)
        else:
            bins["50p"].append(cv)
    return bins

def compute_ecw_coverage(method_name: str, df: pd.DataFrame):
    if df is None or df.empty or "obj_id" not in df or "ttc" not in df:
        return dict(coverage=np.nan, total_tracks=0, covered_tracks=0)
    d = df.copy()
    d = d[d.get("source", "det").isin(["det", "detected", np.nan])]
    d["ttc"] = pd.to_numeric(d["ttc"], errors="coerce")
    d["in_ecw"] = d.get("in_ecw", False).astype(bool)
    pred = _predicted_depth_series(method_name, d)
    d["pred_depth"] = pd.to_numeric(pred, errors="coerce")
    total = 0
    covered = 0
    for oid, g in d.groupby("obj_id"):
        g = g.sort_values("frame_idx") if "frame_idx" in g else g
        t = g["ttc"].dropna().values
        if t.size < 3:
            continue
        diffs = np.diff(t)
        approaching = np.any(diffs < -0.05)
        if not approaching:
            continue
        total += 1
        if g["in_ecw"].any():
            covered += 1
    cov = covered/total if total > 0 else np.nan
    return dict(coverage=cov, total_tracks=total, covered_tracks=covered)

def load_run_summary(out_dir: Path):
    p = out_dir / "run_summary.json"
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read run_summary.json from {out_dir}: {exc}")
        return {}

def load_episodes(out_dir: Path):
    p = out_dir / "episodes.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "run_out_dir" in df.columns:
        df = df[df["run_out_dir"].astype(str) == str(out_dir)]
    return df

def lead_time_stats(ep_df: pd.DataFrame):
    if ep_df is None or ep_df.empty:
        return dict(median=np.nan, p10=np.nan, count=0)
    mask = ep_df.get("warning_issued", False).astype(bool)
    lt = pd.to_numeric(ep_df.loc[mask, "lead_time_s"], errors="coerce").dropna().values
    if lt.size == 0:
        return dict(median=np.nan, p10=np.nan, count=int(mask.sum()))
    median = float(np.median(lt))
    p10 = float(np.percentile(lt, 10))
    return dict(median=median, p10=p10, count=int(mask.sum()))

def compute_alert_rate(ep_df: pd.DataFrame, run_summary: dict):
    if run_summary and run_summary.get("duration_hours") not in (None, 0):
        duration_hours = float(run_summary.get("duration_hours"))
    elif run_summary and run_summary.get("duration_s"):
        duration_hours = float(run_summary.get("duration_s")) / 3600.0
    elif ep_df is not None and not ep_df.empty and "warning_time" in ep_df.columns:
        times = pd.to_numeric(ep_df["warning_time"], errors="coerce").dropna().values
        duration_hours = (times.max() / 3600.0) if times.size else 0.0
    else:
        duration_hours = 0.0
    warnings = 0
    if ep_df is not None and not ep_df.empty and "warning_issued" in ep_df.columns:
        warnings = int(ep_df["warning_issued"].astype(bool).sum())
    return warnings, (warnings / duration_hours) if duration_hours > 0 else np.nan

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

    depth_suffix = base_args.get("depth_backend", "")
    depth_backend = depth_suffix or base_args.get("depth_backend", "")
    tag = slugify(f"{name}-{depth_backend}") if depth_backend else slugify(name)
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
        "--perturb_seed", str(base_args["perturb_seed"]),
        "--frame_sampling", base_args["frame_sampling"],
        "--ecw_depth_threshold", str(base_args["ecw_depth_threshold"]),
        "--lidar_holdout", base_args["lidar_holdout"],
        "--lidar_holdout_ratio", str(base_args["lidar_holdout_ratio"]),
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
    depth_rows = rows
    depth_sample_size = base_args.get("depth_sample_size", 0)
    if depth_sample_size and len(depth_rows) > depth_sample_size:
        rng = np.random.default_rng(base_args.get("perturb_seed", 42))
        picked = rng.choice(len(depth_rows), size=depth_sample_size, replace=False)
        depth_rows = [depth_rows[i] for i in np.sort(picked)]
        print(f"[INFO] Depth metrics using subsample of {len(depth_rows)} frames (seed={base_args.get('perturb_seed', 42)})")
    else:
        depth_rows = depth_rows
    per_range_rows = []
    absrel_values, absrel_groups = [], []
    for frame, lidar_path, mono_path, fused_path in depth_rows:
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
                "DepthBackend": depth_backend,
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
    cv_bins = stratified_ttc_cv(name, obj_df)
    cv_overall = float(np.mean(cv_bins["overall"])) if cv_bins["overall"] else np.nan
    cv_0_25 = float(np.mean(cv_bins["0_25"])) if cv_bins["0_25"] else np.nan
    cv_25_50 = float(np.mean(cv_bins["25_50"])) if cv_bins["25_50"] else np.nan
    cv_50p = float(np.mean(cv_bins["50p"])) if cv_bins["50p"] else np.nan

    episodes_df = load_episodes(out_dir)
    run_summary = load_run_summary(out_dir)
    lt_stats = lead_time_stats(episodes_df)
    warnings_total, alerts_per_hour = compute_alert_rate(episodes_df, run_summary)
    coverage_info = compute_ecw_coverage(name, obj_df)

    summary = dict(
        Method=name,
        DepthBackend=depth_backend,
        AbsRel_mean=mean_absrel,
        AbsRel_CI_lo=lo,
        AbsRel_CI_hi=hi,
        TTC_CV_mean=cv_overall,
        TTC_CV_0_25=cv_0_25,
        TTC_CV_25_50=cv_25_50,
        TTC_CV_50p=cv_50p,
        LeadTime_median_s=lt_stats["median"],
        LeadTime_p10_s=lt_stats["p10"],
        LeadTime_count=lt_stats["count"],
        Warnings_total=warnings_total,
        Alerts_per_hour=alerts_per_hour,
        Coverage_fraction=coverage_info.get("coverage"),
        Coverage_tracks=coverage_info.get("covered_tracks"),
        Approach_tracks=coverage_info.get("total_tracks"),
        Episodes_total=len(episodes_df) if episodes_df is not None else 0,
        **rt,
        OutDir=str(out_dir),
        Precision=warn["Precision"],
        Recall=warn["Recall"],
        F1=warn["F1"],
        Flicker_per1k=warn["Flicker_per1k"]
    )
    if box is not None:
        box["DepthBackend"] = depth_backend

    if episodes_df is not None and not episodes_df.empty:
        ep_subset = episodes_df[[
            "episode_id","class","lead_time_s","warning_issued","onset_ttc","warning_ttc",
            "onset_time","warning_time"
        ]].copy()
        ep_subset["Method"] = name
        ep_subset["DepthBackend"] = depth_backend
    else:
        ep_subset = pd.DataFrame(columns=[
            "episode_id","class","lead_time_s","warning_issued","onset_ttc","warning_ttc",
            "onset_time","warning_time","Method","DepthBackend"
        ])
    return per_range_rows, summary, box, cv_bins, ep_subset

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

def plot_lead_time_cdf(episodes_df, out_png):
    if episodes_df is None or episodes_df.empty:
        print(f"[WARN] Cannot plot lead-time CDF; no episode data for {out_png}")
        return
    plt.figure(figsize=(7,5))
    for method, g in episodes_df.groupby("Method"):
        mask = g.get("warning_issued", False).astype(bool)
        lt = pd.to_numeric(g.loc[mask, "lead_time_s"], errors="coerce").dropna().values
        if lt.size == 0:
            continue
        lt = np.sort(lt)
        y = np.linspace(0, 1, lt.size, endpoint=True)
        plt.plot(lt, y, label=method)
    plt.xlabel("Lead time (s)")
    plt.ylabel("CDF")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)

def plot_hysteresis_tradeoff(hs_df, out_png):
    if hs_df is None or hs_df.empty:
        print(f"[WARN] Cannot plot hysteresis trade-off; no sweep data for {out_png}")
        return
    hs_sorted = hs_df.sort_values("Hysteresis")
    fig, ax1 = plt.subplots(figsize=(7,5))
    ax1.plot(hs_sorted["Hysteresis"], hs_sorted["CV_TTC"], marker='o', color='tab:blue', label="CV(TTC)")
    ax1.set_xlabel("Hysteresis (s)")
    ax1.set_ylabel("CV(TTC) ↓", color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(hs_sorted["Hysteresis"], hs_sorted["F1"], marker='s', linestyle='--', color='tab:red', label="F1")
    ax2.set_ylabel("Proxy F1 ↑", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)

# ----------------------------- Main ------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Run main.py across baselines/ablations and aggregate metrics.")
    ap.add_argument("--main_py", default="main.py")
    ap.add_argument("--out_root", default="experiments")
    ap.add_argument("--paper_out", default="paper_out")
    ap.add_argument("--max_frames", type=int, default=300)
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--lidar_fps", type=float, default=10.0)
    ap.add_argument("--perturb_seed", type=int, default=42,
                    help="Seed forwarded to main.py for deterministic frame sampling and perturbations.")
    ap.add_argument("--frame_sampling", choices=["contiguous","random"], default="contiguous",
                    help="Frame sampling strategy passed to main.py.")
    ap.add_argument("--depth_sample_size", type=int, default=50,
                    help="Number of frames to subsample when computing depth metrics (0=use all).")
    ap.add_argument("--ecw_depth_threshold", type=float, default=7.5,
                    help="Depth (m) defining the ECW bubble / ground-truth boundary.")
    ap.add_argument("--lidar_holdout", choices=["none","beam"], default="beam",
                    help="Disjoint LiDAR split passed to main.py.")
    ap.add_argument("--lidar_holdout_ratio", type=float, default=0.3,
                    help="Hold-out ratio for LiDAR GT split.")
    ap.add_argument("--depth_backend", choices=["midas","zoe","fastdepth","depthanythingv2","monodepth2"], default="midas",
                    help="Primary depth backend when --depth_backends is not provided.")
    ap.add_argument("--depth_backends", nargs="+", default=None,
                    help="Evaluate multiple depth backends sequentially (e.g., --depth_backends midas monodepth2 depthanythingv2).")
    ap.add_argument("--camera_start", type=int, default=15000)
    ap.add_argument("--camera_end",   type=int, default=16500)
    ap.add_argument("--lidar_start",  type=int, default=6000)
    ap.add_argument("--lidar_end",    type=int, default=6600)
    ap.add_argument("--hysteresis", type=float, default=0.5)
    ap.add_argument("--only", nargs="*", default=None, help="Subset of method names to run (exact match).")
    ap.add_argument("--hysteresis_sweep", type=float, nargs="*", default=None,
                    help="Additional hysteresis values to evaluate for the primary method (Ours).")
    args = ap.parse_args()

    if args.depth_backends:
        depth_backend_list = args.depth_backends
    elif any(opt.startswith("--depth_backend") for opt in sys.argv[1:]):
        depth_backend_list = [args.depth_backend]
    else:
        depth_backend_list = ["midas", "monodepth2", "depthanythingv2"]

    main_py   = Path(args.main_py)
    out_root_base  = Path(args.out_root)
    paper_out_base = Path(args.paper_out)

    ensure_dir(out_root_base)
    ensure_dir(paper_out_base)

    frames_cfg_base = dict(
        camera_start=args.camera_start, camera_end=args.camera_end,
        lidar_start=args.lidar_start,   lidar_end=args.lidar_end
    )
    to_run_base = [m for m in METHODS if args.only is None or m[0] in args.only]

    combined_depth_rows = []
    combined_summary_rows = []
    combined_box_rows = []
    combined_episode_rows = []
    combined_ttc_cv_map = {}
    combined_hysteresis_rows = []

    for depth_backend in depth_backend_list:
        print(f"\n=== Depth backend: {depth_backend} ===")
        out_root = out_root_base / depth_backend
        paper_out = paper_out_base / depth_backend
        ensure_dir(out_root)
        ensure_dir(paper_out)

        frames_cfg = frames_cfg_base.copy()
        base_args = dict(
            max_frames=args.max_frames,
            fps=args.fps,
            lidar_fps=args.lidar_fps,
            depth_backend=depth_backend,
            perturb_seed=args.perturb_seed,
            frame_sampling=args.frame_sampling,
            depth_sample_size=max(0, args.depth_sample_size or 0),
            ecw_depth_threshold=args.ecw_depth_threshold,
            lidar_holdout=args.lidar_holdout,
            lidar_holdout_ratio=args.lidar_holdout_ratio
        )

        all_rows = []
        summary_rows = []
        box_rows = []
        episode_rows = []
        ttc_cv_map = {}
        hysteresis_rows = []
        hs_df_backend = pd.DataFrame()

        for name, fusion_mode, ema, mining, sanity, ecw_source in to_run_base:
            per_rows, summary_row, box_df, cv_bins, ep_subset = run_one(
                main_py, out_root, frames_cfg, base_args,
                name, fusion_mode, ema, mining, sanity, ecw_source, args.hysteresis
            )
            all_rows.extend(per_rows)
            summary_rows.append(summary_row)
            if box_df is not None and not box_df.empty:
                box_rows.append(box_df)
            ttc_cv_map[name] = cv_bins.get("overall", [])
            combined_ttc_cv_map[f"{name} [{depth_backend}]"] = cv_bins.get("overall", [])
            if ep_subset is not None and not ep_subset.empty:
                episode_rows.append(ep_subset)

        if args.hysteresis_sweep:
            ours_method = next((m for m in METHODS if m[0] == "Ours (Conf + EMA)"), None)
            if ours_method is None:
                print("[WARN] Hysteresis sweep requested but base method not found in METHODS list.")
            else:
                base_name, fusion_mode, ema, mining, sanity, ecw_source = ours_method
                for h_val in args.hysteresis_sweep:
                    try:
                        h_float = float(h_val)
                    except Exception:
                        print(f"[WARN] Skipping hysteresis value '{h_val}' (not a float)")
                        continue
                    label = f"{base_name} [h={h_float:.2f}]"
                    per_rows_h, summary_h, box_df_h, cv_bins_h, ep_subset_h = run_one(
                        main_py, out_root, frames_cfg, base_args,
                        label, fusion_mode, ema, mining, sanity, ecw_source, h_float
                    )
                    summary_h["Method"] = label
                    summary_h["Hysteresis"] = h_float
                    hysteresis_rows.append(summary_h)

        if episode_rows:
            episodes_all = pd.concat(episode_rows, ignore_index=True)
            episodes_all.to_csv(paper_out/"episodes_aggregated.csv", index=False)
        else:
            episodes_all = pd.DataFrame(columns=["episode_id","class","lead_time_s","warning_issued","onset_ttc","warning_ttc","onset_time","warning_time","Method","DepthBackend"])

        # ---------------- Save tables for this backend ----------------
        depth_df = pd.DataFrame(all_rows)
        depth_df.to_csv(paper_out/"depth_by_range_perframe.csv", index=False)

        if not depth_df.empty:
            depth_df["Seq"] = depth_df["Frame"].apply(seq_from_frame)
            t1_rows = []
            for (method, bin_name), g in depth_df.groupby(["Method","Bin"], sort=False):
                abs_ci_mean, abs_ci_lo, abs_ci_hi = bootstrap_ci(g["AbsRel"].values, g["Seq"].values) if len(g) else (np.nan, np.nan, np.nan)
                rmse_ci_mean, rmse_ci_lo, rmse_ci_hi = bootstrap_ci(g["RMSE"].values, g["Seq"].values) if len(g) else (np.nan, np.nan, np.nan)
                t1_rows.append({
                    "Method": method,
                    "DepthBackend": depth_backend,
                    "Bin": bin_name,
                    "AbsRel_mean": abs_ci_mean,
                    "AbsRel_CI_lo": abs_ci_lo,
                    "AbsRel_CI_hi": abs_ci_hi,
                    "RMSE_mean": rmse_ci_mean,
                    "RMSE_CI_lo": rmse_ci_lo,
                    "RMSE_CI_hi": rmse_ci_hi,
                    "N_pixels": int(g["N"].sum())
                })
            t1 = pd.DataFrame(t1_rows)
        else:
            t1 = pd.DataFrame(columns=["Method","DepthBackend","Bin","AbsRel_mean","AbsRel_CI_lo","AbsRel_CI_hi","RMSE_mean","RMSE_CI_lo","RMSE_CI_hi","N_pixels"])
        t1.to_csv(paper_out/"table1_depth_by_range.csv", index=False)

        tsummary = pd.DataFrame(summary_rows)
        tsummary.to_csv(paper_out/"table_summary_overall.csv", index=False)

        t_proxy = tsummary[[
            "Method","DepthBackend","TTC_CV_mean","TTC_CV_0_25","TTC_CV_25_50","TTC_CV_50p",
            "LeadTime_median_s","LeadTime_p10_s","Flicker_per1k","Alerts_per_hour",
            "Coverage_fraction","Warnings_total"
        ]].copy()
        t_proxy = t_proxy.rename(columns={
            "TTC_CV_mean": "CV_TTC_overall",
            "TTC_CV_0_25": "CV_TTC_0_25",
            "TTC_CV_25_50": "CV_TTC_25_50",
            "TTC_CV_50p": "CV_TTC_50p"
        })
        t_proxy.to_csv(paper_out/"table2_ttc_proxy.csv", index=False)

        t4 = tsummary[["Method","DepthBackend","Det_mean_ms","Depth_mean_ms","Fusion_mean_ms","ECW_mean_ms","Total_mean_ms","Total_95p_ms","FPS"]].copy()
        t4.to_csv(paper_out/"table4_runtime.csv", index=False)

        if box_rows:
            tbox = pd.concat(box_rows, ignore_index=True)
            tbox.to_csv(paper_out/"table_box_mae_by_range.csv", index=False)
        else:
            tbox = pd.DataFrame(columns=["Method","DepthBackend","Bin","MAE","RMSE","Count"])

        twarn = tsummary[["Method","DepthBackend","Precision","Recall","F1","Flicker_per1k"]].copy()
        twarn.to_csv(paper_out/"table2_ecw_metrics.csv", index=False)

        if not t1.empty:
            rmse_overall = t1[t1["Bin"] == "all"][ ["Method","DepthBackend","RMSE_mean"] ].copy()
        else:
            rmse_overall = pd.DataFrame(columns=["Method","DepthBackend","RMSE_mean"])
        ablation_mask = tsummary["Method"].str.contains("Ablation", na=False) | (tsummary["Method"] == "Ours (Conf + EMA)")
        tablation = tsummary[ablation_mask].copy()
        if not tablation.empty:
            if not rmse_overall.empty:
                tablation = tablation.merge(rmse_overall, on=["Method","DepthBackend"], how="left")
            tablation = tablation.rename(columns={
                "Method": "Variant",
                "DepthBackend": "DepthBackend",
                "AbsRel_mean": "AbsRel",
                "RMSE_mean": "RMSE",
                "TTC_CV_mean": "CV_TTC",
                "LeadTime_median_s": "LeadTime_med_s",
                "Flicker_per1k": "Flicker_per_1k"
            })
            tablation = tablation[["Variant","DepthBackend","AbsRel","RMSE","CV_TTC","LeadTime_med_s","Flicker_per_1k","FPS"]]
            tablation.to_csv(paper_out/"table3_ablation.csv", index=False)

        if hysteresis_rows:
            hs_df_backend = pd.DataFrame(hysteresis_rows)
            keep_cols = ["Hysteresis","Method","DepthBackend","CV_TTC","F1","Flicker_per1k","LeadTime_med_s","Alerts_per_hour","FPS","Precision","Recall"]
            if not hs_df_backend.empty:
                if "TTC_CV_mean" in hs_df_backend.columns and "CV_TTC" not in hs_df_backend.columns:
                    hs_df_backend = hs_df_backend.rename(columns={
                        "TTC_CV_mean": "CV_TTC",
                        "Flicker_per1k": "Flicker_per_1k",
                        "LeadTime_median_s": "LeadTime_med_s"
                    })
                for col in keep_cols:
                    if col not in hs_df_backend.columns:
                        hs_df_backend[col] = np.nan
                hs_df_backend = hs_df_backend[keep_cols]
                hs_df_backend.to_csv(paper_out/"table6_hysteresis_sweep.csv", index=False)
        else:
            hs_df_backend = pd.DataFrame(columns=["Hysteresis","Method","DepthBackend","CV_TTC","F1","Flicker_per1k","LeadTime_med_s","Alerts_per_hour","FPS","Precision","Recall"])
            hs_df_backend.to_csv(paper_out/"table6_hysteresis_sweep.csv", index=False)

        try:
            late = tsummary.set_index("Method").loc["Late Fusion (No Conf/EMA)"]
            ours = tsummary.set_index("Method").loc["Ours (Conf + EMA)"]
            imp = {
                "DepthBackend": depth_backend,
                "AbsRel_mean_%improve": 100.0 * (late["AbsRel_mean"] - ours["AbsRel_mean"]) / late["AbsRel_mean"] if pd.notna(late["AbsRel_mean"]) and pd.notna(ours["AbsRel_mean"]) else np.nan,
                "TTC_CV_%improve":      100.0 * (late["TTC_CV_mean"] - ours["TTC_CV_mean"]) / late["TTC_CV_mean"] if pd.notna(late["TTC_CV_mean"]) and pd.notna(ours["TTC_CV_mean"]) else np.nan,
                "F1_%improve":          100.0 * (ours["F1"] - late["F1"]) / (late["F1"]+1e-9) if pd.notna(late["F1"]) and pd.notna(ours["F1"]) else np.nan,
                "FPS_%improve":         100.0 * (ours["FPS"] - late["FPS"]) / (late["FPS"]+1e-9) if pd.notna(late["FPS"]) and pd.notna(ours["FPS"]) else np.nan,
            }
            pd.DataFrame([imp]).to_csv(paper_out/"table_improvement_vs_late.csv", index=False)
        except Exception as e:
            print("[WARN] Could not compute improvement vs Late Fusion:", e)

        plot_absrel_by_range(depth_df, paper_out/"fig_absrel_by_range.png")
        if not tbox.empty:
            plot_box_mae_by_range(tbox, paper_out/"fig_box_mae_by_range.png")
        plot_runtime_stacked(tsummary, paper_out/"fig_runtime_stacked.png")
        plot_flicker_bar(twarn, paper_out/"fig_warning_flicker.png")
        plot_f1_bar(twarn, paper_out/"fig_ecw_f1.png")
        plot_ttc_cv_violin(ttc_cv_map, paper_out/"fig_ttc_cv_violin.png")
        plot_lead_time_cdf(episodes_all, paper_out/"fig_lead_time_cdf.png")
        plot_hysteresis_tradeoff(hs_df_backend, paper_out/"fig_hysteresis_tradeoff.png")

        print("\nDone for backend:", depth_backend)
        print("Tables:")
        for f in [
            "depth_by_range_perframe.csv","table1_depth_by_range.csv",
            "table_summary_overall.csv","table2_ttc_proxy.csv","table3_ablation.csv",
            "table4_runtime.csv","table_box_mae_by_range.csv",
            "table2_ecw_metrics.csv","table_improvement_vs_late.csv",
            "table6_hysteresis_sweep.csv"
        ]:
            print("  -", paper_out/f)
        print("Figures:")
        for f in [
            "fig_absrel_by_range.png","fig_box_mae_by_range.png","fig_runtime_stacked.png",
            "fig_warning_flicker.png","fig_ecw_f1.png","fig_ttc_cv_violin.png",
            "fig_lead_time_cdf.png","fig_hysteresis_tradeoff.png"
        ]:
            print("  -", paper_out/f)

        if not depth_df.empty:
            combined_depth_rows.append(depth_df)
        if not tsummary.empty:
            combined_summary_rows.append(tsummary)
        if not tbox.empty:
            combined_box_rows.append(tbox)
        if not episodes_all.empty:
            combined_episode_rows.append(episodes_all)
        if not hs_df_backend.empty:
            combined_hysteresis_rows.append(hs_df_backend)

    if len(depth_backend_list) > 1:
        combined_out = paper_out_base / "combined"
        ensure_dir(combined_out)

        depth_df_comb = pd.concat(combined_depth_rows, ignore_index=True) if combined_depth_rows else pd.DataFrame()
        tsummary_comb = pd.concat(combined_summary_rows, ignore_index=True) if combined_summary_rows else pd.DataFrame()
        tbox_comb = pd.concat(combined_box_rows, ignore_index=True) if combined_box_rows else pd.DataFrame()
        episodes_comb = pd.concat(combined_episode_rows, ignore_index=True) if combined_episode_rows else pd.DataFrame()
        hs_comb = pd.concat(combined_hysteresis_rows, ignore_index=True) if combined_hysteresis_rows else pd.DataFrame()

        if not depth_df_comb.empty:
            depth_df_comb.to_csv(combined_out/"depth_by_range_perframe.csv", index=False)
            depth_df_plot = depth_df_comb.copy()
            depth_df_plot["Method"] = depth_df_plot["Method"] + " [" + depth_df_plot["DepthBackend"] + "]"
            plot_absrel_by_range(depth_df_plot, combined_out/"fig_absrel_by_range.png")

            depth_df_comb["Seq"] = depth_df_comb["Frame"].apply(seq_from_frame)
            t1_rows = []
            for (method, backend, bin_name), g in depth_df_comb.groupby(["Method","DepthBackend","Bin"], sort=False):
                abs_ci_mean, abs_ci_lo, abs_ci_hi = bootstrap_ci(g["AbsRel"].values, g["Seq"].values) if len(g) else (np.nan, np.nan, np.nan)
                rmse_ci_mean, rmse_ci_lo, rmse_ci_hi = bootstrap_ci(g["RMSE"].values, g["Seq"].values) if len(g) else (np.nan, np.nan, np.nan)
                t1_rows.append({
                    "Method": method,
                    "DepthBackend": backend,
                    "Bin": bin_name,
                    "AbsRel_mean": abs_ci_mean,
                    "AbsRel_CI_lo": abs_ci_lo,
                    "AbsRel_CI_hi": abs_ci_hi,
                    "RMSE_mean": rmse_ci_mean,
                    "RMSE_CI_lo": rmse_ci_lo,
                    "RMSE_CI_hi": rmse_ci_hi,
                    "N_pixels": int(g["N"].sum())
                })
            t1_comb = pd.DataFrame(t1_rows)
            t1_comb.to_csv(combined_out/"table1_depth_by_range.csv", index=False)
        else:
            t1_comb = pd.DataFrame()

        if not tsummary_comb.empty:
            tsummary_comb.to_csv(combined_out/"table_summary_overall.csv", index=False)
            tsummary_plot = tsummary_comb.copy()
            tsummary_plot["Method"] = tsummary_plot["Method"] + " [" + tsummary_plot["DepthBackend"] + "]"
            plot_runtime_stacked(tsummary_plot, combined_out/"fig_runtime_stacked.png")

            t_proxy_comb = tsummary_comb[[
                "Method","DepthBackend","TTC_CV_mean","TTC_CV_0_25","TTC_CV_25_50","TTC_CV_50p",
                "LeadTime_median_s","LeadTime_p10_s","Flicker_per1k","Alerts_per_hour",
                "Coverage_fraction","Warnings_total"
            ]].copy()
            t_proxy_comb = t_proxy_comb.rename(columns={
                "TTC_CV_mean": "CV_TTC_overall",
                "TTC_CV_0_25": "CV_TTC_0_25",
                "TTC_CV_25_50": "CV_TTC_25_50",
                "TTC_CV_50p": "CV_TTC_50p"
            })
            t_proxy_comb.to_csv(combined_out/"table2_ttc_proxy.csv", index=False)

            twarn_comb = tsummary_comb[["Method","DepthBackend","Precision","Recall","F1","Flicker_per1k"]].copy()
            twarn_comb.to_csv(combined_out/"table2_ecw_metrics.csv", index=False)
            twarn_plot = twarn_comb.copy()
            twarn_plot["Method"] = twarn_plot["Method"] + " [" + twarn_plot["DepthBackend"] + "]"
            plot_flicker_bar(twarn_plot, combined_out/"fig_warning_flicker.png")
            plot_f1_bar(twarn_plot, combined_out/"fig_ecw_f1.png")

            t4_comb = tsummary_comb[["Method","DepthBackend","Det_mean_ms","Depth_mean_ms","Fusion_mean_ms","ECW_mean_ms","Total_mean_ms","Total_95p_ms","FPS"]].copy()
            t4_comb.to_csv(combined_out/"table4_runtime.csv", index=False)

            ablation_mask_comb = tsummary_comb["Method"].str.contains("Ablation", na=False) | (tsummary_comb["Method"] == "Ours (Conf + EMA)")
            tablation_comb = tsummary_comb[ablation_mask_comb].copy()
            if not tablation_comb.empty and not t1_comb.empty:
                rmse_all = t1_comb[t1_comb["Bin"] == "all"][ ["Method","DepthBackend","RMSE_mean"] ]
                tablation_comb = tablation_comb.merge(rmse_all, on=["Method","DepthBackend"], how="left")
                tablation_comb = tablation_comb.rename(columns={
                    "Method": "Variant",
                    "AbsRel_mean": "AbsRel",
                    "RMSE_mean": "RMSE",
                    "TTC_CV_mean": "CV_TTC",
                    "LeadTime_median_s": "LeadTime_med_s",
                    "Flicker_per1k": "Flicker_per_1k"
                })
                tablation_comb = tablation_comb[["Variant","DepthBackend","AbsRel","RMSE","CV_TTC","LeadTime_med_s","Flicker_per_1k","FPS"]]
                tablation_comb.to_csv(combined_out/"table3_ablation.csv", index=False)

        if not tbox_comb.empty:
            tbox_comb.to_csv(combined_out/"table_box_mae_by_range.csv", index=False)
            tbox_plot = tbox_comb.copy()
            tbox_plot["Method"] = tbox_plot["Method"] + " [" + tbox_plot["DepthBackend"] + "]"
            plot_box_mae_by_range(tbox_plot, combined_out/"fig_box_mae_by_range.png")

        if not episodes_comb.empty:
            episodes_comb.to_csv(combined_out/"episodes_aggregated.csv", index=False)
            episodes_plot = episodes_comb.copy()
            episodes_plot["Method"] = episodes_plot["Method"] + " [" + episodes_plot["DepthBackend"] + "]"
            plot_lead_time_cdf(episodes_plot, combined_out/"fig_lead_time_cdf.png")

        if not hs_comb.empty:
            hs_comb.to_csv(combined_out/"table6_hysteresis_sweep.csv", index=False)

        if combined_ttc_cv_map:
            plot_ttc_cv_violin(combined_ttc_cv_map, combined_out/"fig_ttc_cv_violin.png")

        print("\nCombined results saved under:", combined_out)

if __name__ == "__main__":
    main()
