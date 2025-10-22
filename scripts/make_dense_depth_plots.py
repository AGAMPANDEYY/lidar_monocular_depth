#!/usr/bin/env python3
"""Generate dense-depth tables/figures for Section 6.1 from saved CSV outputs."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DISPLAY_NAMES = {
    'midas': 'MiDaS',
    'monodepth2': 'MonoDepth2',
    'depthanythingv2': 'Depth-Anything-V2'
}


def backend_display_name(table: pd.DataFrame):
    if 'DepthBackend' not in table.columns:
        return None
    values = table['DepthBackend'].dropna().unique()
    if len(values) == 0:
        return None
    names = [DISPLAY_NAMES.get(str(v).lower(), str(v)) for v in values]
    if len(names) == 1:
        return names[0]
    return '(' + ' + '.join(names) + ')'


def base_method_label(name: str) -> str:
    if not isinstance(name, str):
        return ''
    base = name.split(' [')[0]
    if base.startswith('Ablation'):
        return 'Ablations (No EMA / No Mining / No Sanity Checks)'
    return base


def format_method_label(row):
    backend = row.get('DepthBackend')
    base = base_method_label(row.get('Method'))
    if pd.notna(backend) and str(backend) != '':
        label = DISPLAY_NAMES.get(str(backend).lower(), str(backend))
        return f"{base} [{label}]"
    return base


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def print_table(table: pd.DataFrame, title: str = None):
    if title:
        print(f"\n{title}")
    if table.empty:
        print("[WARN] table is empty")
    else:
        print(table.to_string(index=False))


def plot_absrel_by_range(table: pd.DataFrame, out_png: Path):
    if table.empty:
        print(f"[WARN] Cannot plot {out_png}: table is empty")
        return

    table = table.copy()
    combined = 'DepthBackend' in table.columns and table['DepthBackend'].nunique() > 1

    if combined:
        backends = list(table['DepthBackend'].dropna().unique())
        if not backends:
            backends = [None]
        n_cols = len(backends)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.0*n_cols, 4), sharey=True)
        if n_cols == 1:
            axes = [axes]
        bins = ["0_10", "10_25", "25_50", "50p"]
        for ax, backend in zip(axes, backends):
            sub = table[table['DepthBackend'] == backend].copy()
            sub['DisplayMethod'] = sub['Method'].apply(base_method_label)
            methods = sub['DisplayMethod'].unique()
            width = 0.2
            x = list(range(len(bins)))
            for i, method in enumerate(methods):
                subset = sub[sub['DisplayMethod'] == method]
                heights = [
                    subset[subset['Bin'] == b]['AbsRel_mean'].values[0]
                    if not subset[subset['Bin'] == b].empty else float('nan')
                    for b in bins
                ]
                offsets = [xi + (i - len(methods)/2)*width + width/2 for xi in x]
                ax.bar(offsets, heights, width=width, label=method)
            ymax = sub['AbsRel_mean'].max()
            if pd.notna(ymax):
                ax.set_ylim(0, ymax * 1.3)
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(["0–10", "10–25", "25–50", ">50"])
            ax.set_xlabel("Range (m)")
            if ax is axes[0]:
                ax.set_ylabel("AbsRel ↓")
            backend_name = DISPLAY_NAMES.get(str(backend).lower(), str(backend)) if backend is not None else ""
            ax.set_title(backend_name)
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
        fig.suptitle("AbsRel by Range", y=0.98)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches='tight', bbox_extra_artists=(legend,))
        plt.close(fig)
    else:
        if 'DepthBackend' in table.columns:
            table['DisplayMethod'] = table.apply(format_method_label, axis=1)
        else:
            table['DisplayMethod'] = table['Method']

        plt.figure(figsize=(7.5, 4))
        bins = ["0_10", "10_25", "25_50", "50p"]
        methods = table["DisplayMethod"].unique()
        width = 0.18
        x = list(range(len(bins)))

        for i, method in enumerate(methods):
            sub = table[table["DisplayMethod"] == method]
            heights = [
                sub[sub["Bin"] == b]["AbsRel_mean"].values[0]
                if not sub[sub["Bin"] == b].empty else float("nan")
                for b in bins
            ]
            offsets = [xi + (i - len(methods)/2)*width + width/2 for xi in x]
            plt.bar(offsets, heights, width=width, label=method)
        ymax = table['AbsRel_mean'].max()
        if pd.notna(ymax):
            plt.ylim(0, ymax * 1.3)

        plt.xticks(range(len(bins)), ["0–10", "10–25", "25–50", ">50"])
        plt.xlabel("Range (m)")
        plt.ylabel("AbsRel ↓")
        title = "AbsRel by Range"
        backend_label = backend_display_name(table)
        if backend_label:
            title += f" {backend_label}"
        plt.title(title, y=1.05)
        legend = plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches='tight', bbox_extra_artists=(legend,))
        plt.close()


def plot_method_absrel_per_bin(table: pd.DataFrame, out_dir: Path, suffix: str):
    if table.empty:
        print("[WARN] Cannot plot per-bin method charts: table is empty")
        return

    method_order = [
        "Monocular-Only (Scaled)",
        "Late Fusion (No Conf/EMA)",
        "Ours (Conf + EMA)"
    ]
    bin_labels = {
        "all": "All distances",
        "0_10": "0–10 m",
        "10_25": "10–25 m",
        "25_50": "25–50 m",
        "50p": ">50 m"
    }
    backend_order_keys = ["midas", "monodepth2", "depthanythingv2"]

    df = table.copy()
    df = df[df["Method"].isin(method_order)]
    if df.empty:
        print("[WARN] No rows matching requested methods for per-bin charts.")
        return

    df["BackendKey"] = df["DepthBackend"].astype(str).str.lower()
    df["BackendLabel"] = df["BackendKey"].map(DISPLAY_NAMES).fillna(df["DepthBackend"].astype(str))

    available_backend_keys = [b for b in backend_order_keys if b in df["BackendKey"].unique()]
    if not available_backend_keys:
        available_backend_keys = sorted(df["BackendKey"].unique())

    bins_order = [b for b in ["all", "0_10", "10_25", "25_50", "50p"] if b in df["Bin"].unique()]

    for bin_name in bins_order:
        sub = df[df["Bin"] == bin_name]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4))
        x = np.arange(len(available_backend_keys))
        width = 0.22

        for idx, method in enumerate(method_order):
            method_rows = sub[sub["Method"] == method]
            heights = []
            for backend in available_backend_keys:
                val = method_rows[method_rows["BackendKey"] == backend]["AbsRel_mean"]
                heights.append(val.iloc[0] if not val.empty else np.nan)
            offset = x + (idx - len(method_order)/2) * width + width/2
            ax.bar(offset, heights, width=width, label=method)

        backend_labels = [
            DISPLAY_NAMES.get(key, key) for key in available_backend_keys
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(backend_labels)
        ax.set_ylabel("AbsRel ↓")
        ax.set_xlabel("Depth backend")
        pretty_bin = bin_labels.get(bin_name, bin_name)
        ax.set_title(f"AbsRel by Method – {pretty_bin} ({suffix})")
        legend = ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
        fig.tight_layout()
        out_path = out_dir / f"fig_absrel_{bin_name}_{suffix}.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight', bbox_extra_artists=(legend,))
        plt.close(fig)


def plot_box_mae_by_range(table: pd.DataFrame, out_png: Path):
    if table.empty:
        print(f"[WARN] No box MAE data for {out_png}")
        return

    table = table.copy()
    combined = 'DepthBackend' in table.columns and table['DepthBackend'].nunique() > 1

    if combined:
        backends = list(table['DepthBackend'].dropna().unique())
        if not backends:
            backends = [None]
        n_cols = len(backends)
        fig, axes = plt.subplots(1, n_cols, figsize=(5.0*n_cols, 4), sharey=True)
        if n_cols == 1:
            axes = [axes]
        bins = ["0_10", "10_25", "25_50", "50p"]
        for ax, backend in zip(axes, backends):
            sub = table[table['DepthBackend'] == backend].copy()
            sub['DisplayMethod'] = sub['Method'].apply(base_method_label)
            methods = sub['DisplayMethod'].unique()
            width = 0.2
            x = list(range(len(bins)))
            for i, method in enumerate(methods):
                subset = sub[sub['DisplayMethod'] == method]
                heights = [
                    subset[subset['Bin'] == b]['MAE'].values[0]
                    if not subset[subset['Bin'] == b].empty else float('nan')
                    for b in bins
                ]
                offsets = [xi + (i - len(methods)/2)*width + width/2 for xi in x]
                ax.bar(offsets, heights, width=width, label=method)
            ymax = sub['MAE'].max()
            if pd.notna(ymax):
                ax.set_ylim(0, ymax * 1.3)
            ax.set_xticks(range(len(bins)))
            ax.set_xticklabels(["0–10", "10–25", "25–50", ">50"])
            ax.set_xlabel("Range (m)")
            if ax is axes[0]:
                ax.set_ylabel("Box MAE (m) ↓")
            backend_name = DISPLAY_NAMES.get(str(backend).lower(), str(backend)) if backend is not None else ""
            ax.set_title(backend_name)
        handles, labels = axes[0].get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
        fig.suptitle("Box MAE by Range", y=0.98)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches='tight', bbox_extra_artists=(legend,))
        plt.close(fig)
    else:
        if 'DepthBackend' in table.columns:
            table['DisplayMethod'] = table.apply(format_method_label, axis=1)
        else:
            table['DisplayMethod'] = table['Method']

        plt.figure(figsize=(7.5, 4))
        bins = ["0_10", "10_25", "25_50", "50p"]
        methods = table["DisplayMethod"].unique()
        width = 0.18

        for i, method in enumerate(methods):
            sub = table[table["DisplayMethod"] == method]
            heights = [
                sub[sub["Bin"] == b]["MAE"].values[0]
                if not sub[sub["Bin"] == b].empty else float("nan")
                for b in bins
            ]
            offsets = [xi + (i - len(methods)/2)*width + width/2 for xi in range(len(bins))]
            plt.bar(offsets, heights, width=width, label=method)
        ymax = table['MAE'].max()
        if pd.notna(ymax):
            plt.ylim(0, ymax * 1.3)

        plt.xticks(range(len(bins)), ["0–10", "10–25", "25–50", ">50"])
        plt.xlabel("Range (m)")
        plt.ylabel("Box MAE (m) ↓")
        title = "Box MAE by Range"
        backend_label = backend_display_name(table)
        if backend_label:
            title += f" {backend_label}"
        plt.title(title, y=1.05)
        legend = plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300, bbox_inches='tight', bbox_extra_artists=(legend,))
        plt.close()


def export_box_mae_summary(table: pd.DataFrame, out_dir: Path, label: str, title_suffix: str, is_combined: bool = False):
    if table.empty:
        return
    mapping = {
        "0_10": "0–10 m",
        "10_25": "10–25 m",
        "25_50": "25–50 m",
        "50p": ">50 m"
    }
    if not is_combined:
        table = table.copy()
        if 'DepthBackend' in table.columns:
            table['DisplayMethod'] = table.apply(format_method_label, axis=1)
        else:
            table['DisplayMethod'] = table['Method']
        # Consolidate duplicate method/bin entries (e.g., multiple ablations collapsing to one label).
        agg_dict = {'MAE': 'mean'}
        if 'Count' in table.columns:
            agg_dict['Count'] = 'sum'
        table = table.groupby(['DisplayMethod', 'Bin'], as_index=False).agg(agg_dict)
        summary = table.pivot(index='DisplayMethod', columns='Bin', values='MAE')
        summary = summary.rename(columns=mapping)
        summary = summary.reset_index().rename(columns={'DisplayMethod': 'Method'})
        summary = summary.round(2)
        summary_path = out_dir / f"table_box_mae_summary_{label}.csv"
        summary.to_csv(summary_path, index=False)
        print_table(summary, f"Box MAE by Range Summary {title_suffix}")
    else:
        # Condensed for combined
        box_summary = table.copy()
        box_summary['MethodLabel'] = box_summary.apply(format_method_label, axis=1)
        box_summary["Backend"] = box_summary["MethodLabel"].str.extract(r"\[(.*)\]")
        box_summary["Backend"] = box_summary["Backend"].str.lower().map(DISPLAY_NAMES).fillna(box_summary["Backend"])
        box_summary["BaseMethod"] = box_summary["MethodLabel"].str.replace(r" \[.*\]", "", regex=True)
        box_summary.loc[box_summary["BaseMethod"].str.startswith("Ablation"), "BaseMethod"] = "Ablations (No EMA / No Mining / No Sanity Checks)"
        box_summary.loc[box_summary["BaseMethod"] == "Ours (Conf + EMA)", "BaseMethod"] = "**Ours (Conf + EMA)**"
        box_summary.loc[box_summary["BaseMethod"] == "Monocular-Only (Scaled)", "BaseMethod"] = "Monocular-Only (Scaled)"

        # Special handling for LiDAR-Only
        lidar_mask_box = box_summary["BaseMethod"].str.startswith("LiDAR-Only")
        lidar_agg_box = box_summary[lidar_mask_box].groupby(["BaseMethod", "Bin"]).agg({"MAE": "mean"}).reset_index()
        lidar_agg_box["Backend"] = "Reference"
        other_box = box_summary[~lidar_mask_box]
        other_agg_box = other_box.groupby(["BaseMethod", "Backend", "Bin"]).agg({"MAE": "mean"}).reset_index()
        all_agg_box = pd.concat([other_agg_box, lidar_agg_box])

        # Pivot with multi-level columns: Backend and Bin
        pivot_box = all_agg_box.pivot(index="BaseMethod", columns=["Backend", "Bin"], values="MAE")

        # Flatten columns
        pivot_box.columns = [f"{col[0]} {mapping.get(col[1], col[1])} MAE ↓" for col in pivot_box.columns]

        # Round numerics to 2 decimal places
        numeric_cols = pivot_box.select_dtypes(include='number').columns
        pivot_box[numeric_cols] = pivot_box[numeric_cols].round(2)

        # Fill NaN with '-'
        pivot_box = pivot_box.fillna('-')

        # Reorder rows
        rows_order = [
            "Monocular-Only (Scaled)",
            "Late Fusion (No Conf/EMA)",
            "**Ours (Conf + EMA)**",
            "Ablations (No EMA / No Mining / No Sanity Checks)",
            "LiDAR-Only (Projected)"
        ]
        pivot_box = pivot_box.reindex(rows_order).reset_index().rename(columns={"BaseMethod": "Method"})

        box_summary_path = out_dir / "table_box_mae_summary_combined.csv"
        pivot_box.to_csv(box_summary_path, index=False)
        print_table(pivot_box, "Condensed Box MAE Summary (Combined)")
        print("\nAdd footnotes: (*) LiDAR-Only is a reference upper bound; (**) Ablations identical to Ours in this dataset.")


def main():
    parser = argparse.ArgumentParser(description="Generate dense-depth plots/tables from saved CSV outputs.")
    parser.add_argument("--paper_out", default="paper_out", help="Root directory containing per-backend outputs.")
    parser.add_argument("--backend", default=None,
                        help="Specific backend (e.g., midas). If omitted, use the combined bundle.")
    parser.add_argument("--output", default="paper_figures",
                        help="Directory to store generated figures/tables.")
    args = parser.parse_args()

    paper_out = Path(args.paper_out)
    backend_dir = paper_out / args.backend if args.backend else paper_out / "combined"
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_overall = None
    is_combined = args.backend is None

    # Table 1: dense depth by range
    t1_path = backend_dir / "table1_depth_by_range.csv"
    t1 = load_table(t1_path)
    title = f"Table 1 ({args.backend})" if args.backend else "Table 1 (Combined)"
    print_table(t1, title)
    absrel_plot = out_dir / (f"fig_absrel_by_range_{args.backend or 'combined'}.png")
    plot_absrel_by_range(t1, absrel_plot)
    if args.backend:
        suffix_label = DISPLAY_NAMES.get(args.backend.lower(), args.backend)
    else:
        suffix_label = "Combined"
    plot_method_absrel_per_bin(t1, out_dir, suffix_label)

    overall = t1[t1["Bin"] == "all"].copy()
    if not overall.empty:
        overall["MethodLabel"] = overall.apply(format_method_label, axis=1)
        overall_summary = pd.DataFrame({
            "Method": overall["MethodLabel"],
            "AbsRel": overall["AbsRel_mean"],
            "AbsRel CI": overall.apply(lambda r: f"[{r['AbsRel_CI_lo']:.3f}, {r['AbsRel_CI_hi']:.3f}]", axis=1),
            "RMSE": overall["RMSE_mean"],
            "RMSE CI": overall.apply(lambda r: f"[{r['RMSE_CI_lo']:.3f}, {r['RMSE_CI_hi']:.3f}]", axis=1),
            "Pixels": overall["N_pixels"].astype(int)
        })
        overall_title = "Overall Dense Depth"
        label = backend_display_name(overall)
        if label:
            overall_title += f" {label}"
        print_table(overall_summary, overall_title)
        overall_path = out_dir / (f"table_depth_overall_{args.backend or 'combined'}.csv")
        overall_summary.to_csv(overall_path, index=False)
        if is_combined:
            combined_overall = overall_summary.copy()

    # Box MAE table + plot (supplementary)
    box_path = backend_dir / "table_box_mae_by_range.csv"
    if box_path.exists():
        tbox = load_table(box_path)
        title_box = "Table S-1" + (f" ({args.backend})" if args.backend else " (Combined)")
        print_table(tbox, title_box)
        box_plot = out_dir / (f"fig_box_mae_by_range_{args.backend or 'combined'}.png")
        plot_box_mae_by_range(tbox, box_plot)
        export_box_mae_summary(tbox, out_dir, args.backend or 'combined', f"({args.backend})" if args.backend else "(Combined)", is_combined=is_combined)

    if is_combined and combined_overall is not None:
        summary = combined_overall.copy()
        summary["Backend"] = summary["Method"].str.extract(r"\[(.*)\]")
        summary["Backend"] = summary["Backend"].str.lower().map(DISPLAY_NAMES).fillna(summary["Backend"])
        summary["BaseMethod"] = summary["Method"].str.replace(r" \[.*\]", "", regex=True)
        summary.loc[summary["BaseMethod"].str.startswith("Ablation"), "BaseMethod"] = "Ablations (No EMA / No Mining / No Sanity Checks)"
        summary.loc[summary["BaseMethod"] == "Ours (Conf + EMA)", "BaseMethod"] = "**Ours (Conf + EMA)**"
        summary.loc[summary["BaseMethod"] == "Monocular-Only (Scaled)", "BaseMethod"] = "Monocular-Only (Scaled)"

        # Special handling for LiDAR-Only
        lidar_mask = summary["BaseMethod"].str.startswith("LiDAR-Only")
        lidar_agg = summary[lidar_mask].groupby("BaseMethod").agg({"AbsRel": "mean", "RMSE": "mean"}).reset_index()
        lidar_agg["Backend"] = "Reference"
        other_summary = summary[~lidar_mask]
        other_agg = other_summary.groupby(["BaseMethod", "Backend"]).agg({"AbsRel": "mean", "RMSE": "mean"}).reset_index()
        all_agg = pd.concat([other_agg, lidar_agg])

        pivot = all_agg.pivot(index="BaseMethod", columns="Backend", values=["AbsRel", "RMSE"])
        pivot = pivot.reorder_levels([1, 0], axis=1)
        pivot = pivot.sort_index(axis=1, level=0)

        for col in pivot.columns:
            if col[1] == 'AbsRel':
                pivot[col] = pivot[col].round(3)
            elif col[1] == 'RMSE':
                pivot[col] = pivot[col].round(2)

        pivot = pivot.fillna('-')

        pivot.columns = [f"{col[0]} {'AbsRel ↓' if col[1]=='AbsRel' else 'RMSE (m) ↓'}" for col in pivot.columns]

        # Reorder rows
        rows_order = [
            "Monocular-Only (Scaled)",
            "Late Fusion (No Conf/EMA)",
            "**Ours (Conf + EMA)**",
            "Ablations (No EMA / No Mining / No Sanity Checks)",
            "LiDAR-Only (Projected)"
        ]
        pivot = pivot.reindex(rows_order).reset_index().rename(columns={"BaseMethod": "Method"})

        summary_path = out_dir / "table_depth_summary_combined.csv"
        pivot.to_csv(summary_path, index=False)
        print_table(pivot, "Condensed Dense-Depth Summary (Combined)")
        print("\nAdd footnotes: (*) LiDAR-Only is a reference upper bound; (**) Ablations identical to Ours in this dataset.")

    print(f"\nArtifacts written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
