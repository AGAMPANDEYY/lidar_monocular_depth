#!/usr/bin/env python3
"""Generate a single binned histogram of AbsRel by distance bin for specified methods across backends."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

DISPLAY_NAMES = {
    'midas': 'MiDaS',
    'monodepth2': 'MonoDepth2',
    'depthanythingv2': 'Depth-Anything-V2'
}

def base_method_label(name: str) -> str:
    if not isinstance(name, str):
        return ''
    base = name.split(' [')[0]
    return base

def lighten_color(color, amount=0.5):
    """Lighten a color by blending with white."""
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return mcolors.to_hex((1 - amount) * np.array(c) + amount * np.array([1, 1, 1]))

BASE_COLORS = {
    "Monocular-Only (Scaled)": '#1f77b4',  # Blue
    "Late Fusion (No Conf/EMA)": '#ff7f0e',  # Orange
    "Ours (Conf + EMA)": '#2ca02c'  # Green
}

def build_color_gradients():
    gradients = {}
    for method, color in BASE_COLORS.items():
        gradients[method] = [
            color,
            lighten_color(color, 0.3),
            lighten_color(color, 0.6)
        ]
    return gradients

COLOR_GRADIENTS = build_color_gradients()

def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def plot_absrel_by_bin_combined(table: pd.DataFrame, out_png: Path):
    if table.empty:
        print(f"[WARN] Cannot plot {out_png}: table is empty")
        return

    # Filter to specified methods
    method_order = [
        "Monocular-Only (Scaled)",
        "Late Fusion (No Conf/EMA)",
        "Ours (Conf + EMA)"
    ]
    df = table[table["Method"].isin(method_order)].copy()
    if df.empty:
        print("[WARN] No rows matching requested methods for combined bin plot.")
        return

    # Define bins and their labels
    bins = ["all", "0_10", "10_25", "25_50", "50p"]
    bin_labels = {
        "all": "All",
        "0_10": "0–10 m",
        "10_25": "10–25 m",
        "25_50": "25–50 m",
        "50p": ">50 m"
    }
    backends = ["midas", "monodepth2", "depthanythingv2"]
    backend_labels = [DISPLAY_NAMES.get(b, b) for b in backends]

    # Create single figure
    fig, ax = plt.subplots(figsize=(14, 4))
    width = 0.08  # Width of each bar
    n_methods = len(method_order)
    n_backends = len(backends)
    group_width = n_methods * n_backends * width + width  # Total width per bin group
    x = np.arange(len(bins)) * (group_width + 0.2)  # Centers of each bin group

    # Plot bars
    for i, bin_name in enumerate(bins):
        sub = df[df["Bin"] == bin_name]
        if sub.empty:
            continue
        for j, method in enumerate(method_order):
            method_rows = sub[sub["Method"] == method]
            for k, backend in enumerate(backends):
                val = method_rows[method_rows["DepthBackend"] == backend]["AbsRel_mean"]
                height = val.iloc[0] if not val.empty else np.nan
                offset = x[i] + (j * n_backends + k - (n_methods * n_backends - 1) / 2) * width
                ax.bar(offset, height, width=width, color=COLOR_GRADIENTS[method][k],
                       label=f"{method} [{backend_labels[k]}]" if i == 0 else None)

    # Customize plot
    ax.set_xticks(x)
    ax.set_xticklabels([bin_labels.get(b, b) for b in bins])
    ax.set_xlabel("Distance Bin")
    ax.set_ylabel("AbsRel ↓")
    ax.set_title("AbsRel by Distance Bin Across Methods and Backends")

    # Set y-axis limit
    ymax = df['AbsRel_mean'].max()
    if pd.notna(ymax):
        ax.set_ylim(0, ymax * 1.3)

    # Add legend outside the plot, further to the right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:n_methods * n_backends], labels[:n_methods * n_backends],
              loc='center right', bbox_to_anchor=(1.35, 0.5), frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_box_mae_by_bin_combined(table: pd.DataFrame, out_png: Path):
    if table.empty:
        print(f"[WARN] Cannot plot {out_png}: table is empty")
        return

    method_order = [
        "Monocular-Only (Scaled)",
        "Late Fusion (No Conf/EMA)",
        "Ours (Conf + EMA)"
    ]
    bins = ["0_10", "10_25", "25_50", "50p"]
    bin_labels = {
        "0_10": "0–10 m",
        "10_25": "10–25 m",
        "25_50": "25–50 m",
        "50p": ">50 m"
    }
    backend_keys = ["midas", "monodepth2", "depthanythingv2"]

    df = table.copy()
    df = df[df["Method"].isin(method_order)]
    if df.empty:
        print("[WARN] No rows matching requested methods for combined box-MAE plot.")
        return
    df["DepthBackend"] = df["DepthBackend"].astype(str).str.lower()
    df["MAE"] = pd.to_numeric(df["MAE"], errors="coerce")

    available_backends = [b for b in backend_keys if b in df["DepthBackend"].unique()]
    if not available_backends:
        available_backends = sorted(df["DepthBackend"].unique())

    fig, ax = plt.subplots(figsize=(14, 4))
    width = 0.08
    n_methods = len(method_order)
    n_backends = len(available_backends)
    group_width = n_methods * n_backends * width + width
    x = np.arange(len(bins)) * (group_width + 0.2)

    for i, bin_name in enumerate(bins):
        sub = df[df["Bin"] == bin_name]
        if sub.empty:
            continue
        for j, method in enumerate(method_order):
            method_rows = sub[sub["Method"] == method]
            for k, backend in enumerate(available_backends):
                val = method_rows[method_rows["DepthBackend"] == backend]["MAE"]
                height = val.iloc[0] if not val.empty else np.nan
                offset = x[i] + (j * n_backends + k - (n_methods * n_backends - 1) / 2) * width
                if backend in backend_keys:
                    color = COLOR_GRADIENTS[method][backend_keys.index(backend)]
                else:
                    color = BASE_COLORS[method]
                ax.bar(offset, height, width=width, color=color,
                       label=f"{method} [{DISPLAY_NAMES.get(backend, backend)}]" if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels([bin_labels.get(b, b) for b in bins])
    ax.set_xlabel("Distance Bin")
    ax.set_ylabel("Box MAE (m) ↓")
    ax.set_title("Box MAE by Distance Bin Across Methods and Backends")

    ymax = df["MAE"].max()
    if pd.notna(ymax):
        ax.set_ylim(0, ymax * 1.3)

    handles, labels = ax.get_legend_handles_labels()
    max_labels = min(len(handles), n_methods * n_backends)
    ax.legend(handles[:max_labels], labels[:max_labels],
              loc='center right', bbox_to_anchor=(1.35, 0.5), frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Generate AbsRel binned histogram across backends.")
    parser.add_argument("--paper_out", default="paper_out", help="Root directory containing per-backend outputs.")
    parser.add_argument("--output", default="paper_figures", help="Directory to store generated figure.")
    args = parser.parse_args()

    paper_out = Path(args.paper_out)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load combined tables
    t1_path = paper_out / "combined" / "table1_depth_by_range.csv"
    t1 = load_table(t1_path)
    box_path = paper_out / "combined" / "table_box_mae_by_range.csv"
    tbox = load_table(box_path)
    
    # Generate combined plot
    absrel_plot = out_dir / "fig_absrel_by_bin_combined.png"
    plot_absrel_by_bin_combined(t1, absrel_plot)
    box_plot = out_dir / "fig_box_mae_by_bin_combined.png"
    plot_box_mae_by_bin_combined(tbox, box_plot)
    
    print(f"\nFigures written to:\n  {absrel_plot.resolve()}\n  {box_plot.resolve()}")

if __name__ == "__main__":
    main()
