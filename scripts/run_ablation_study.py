#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Eval.ablations import AblationVariant, evaluate_variant
import numpy as np

def create_variants():
    """Create all ablation study variants"""
    variants = []
    
    # Baseline (all features enabled)
    baseline = AblationVariant(
        name="Full System",
        use_confidence=True,
        use_ema=True,
        use_mining=True
    )
    variants.append(baseline)
    
    # Individual ablations
    variants.extend([
        AblationVariant(
            name="No Confidence",
            use_confidence=False,
            use_ema=True,
            use_mining=True
        ),
        AblationVariant(
            name="No Temporal",
            use_confidence=True,
            use_ema=False,
            use_mining=True
        ),
        AblationVariant(
            name="No Mining",
            use_confidence=True,
            use_ema=True,
            use_mining=False
        )
    ])
    
    return variants

def plot_results(results_df, output_dir):
    """Generate plots for the ablation study results"""
    # Set style
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Depth Accuracy Plot
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    variants = results_df['Variant']
    abs_rel = results_df['abs_rel_mean']
    abs_rel_ci = results_df['abs_rel_ci']
    
    bars = plt.bar(variants, abs_rel, yerr=abs_rel_ci, capsize=5)
    plt.xticks(rotation=45, ha='right')
    plt.title('Depth Estimation Accuracy (Abs Rel)')
    plt.ylabel('Absolute Relative Error')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'depth_accuracy.png'))
    plt.close()
    
    # 2. Warning Performance Plot
    plt.figure(figsize=(10, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 Score
    bars1 = ax1.bar(variants, results_df['f1_mean'], yerr=results_df['f1_ci'], capsize=5)
    ax1.set_title('Warning F1 Score')
    ax1.set_ylabel('F1 Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Flicker Rate
    bars2 = ax2.bar(variants, results_df['flicker'], capsize=5)
    ax2.set_title('Warning Flicker Rate')
    ax2.set_ylabel('Flicker Rate (per minute)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bars, ax in [(bars1, ax1), (bars2, ax2)]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'warning_performance.png'))
    plt.close()

def create_latex_table(results_df):
    """Generate LaTeX table for the paper"""
    latex_table = """\\begin{table}[t]
\\centering
\\begin{tabular}{l|cccc|c}
\\toprule
Variant & Abs Rel & RMSE & F1 Score & TTC CV & FPS \\\\
\\midrule
"""
    
    for _, row in results_df.iterrows():
        latex_table += f"{row['Variant']} & "
        latex_table += f"{row['abs_rel_mean']:.3f} ± {row['abs_rel_ci']:.3f} & "
        latex_table += f"{row['rmse_mean']:.2f} ± {row['rmse_ci']:.2f} & "
        latex_table += f"{row['f1_mean']:.3f} ± {row['f1_ci']:.3f} & "
        latex_table += f"{row['cv_ttc_mean']:.3f} ± {row['cv_ttc_ci']:.3f} & "
        latex_table += f"{row['fps']:.1f} \\\\\n"
    
    latex_table += """\\bottomrule
\\end{tabular}
\\caption{Ablation study results comparing different system components.}
\\label{tab:ablation}
\\end{table}"""
    
    return latex_table

def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--data_dir', required=True, help='Directory containing fusion results')
    parser.add_argument('--output_dir', default='results/ablation', help='Output directory for results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run ablation studies
    variants = create_variants()
    results = []
    
    for variant in variants:
        print(f"Evaluating variant: {variant.name}")
        metrics = evaluate_variant(variant, args.data_dir)
        metrics['Variant'] = variant.name
        results.append(metrics)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save raw results
    results_df.to_csv(os.path.join(args.output_dir, 'ablation_results.csv'), index=False)
    
    # Generate plots
    plot_results(results_df, args.output_dir)
    
    # Generate LaTeX table
    latex_table = create_latex_table(results_df)
    with open(os.path.join(args.output_dir, 'ablation_table.tex'), 'w') as f:
        f.write(latex_table)
    
    print(f"\nResults saved to {args.output_dir}")
    print("Generated files:")
    print("- ablation_results.csv: Raw results in CSV format")
    print("- plots/depth_accuracy.png: Depth estimation accuracy plot")
    print("- plots/warning_performance.png: Warning performance plots")
    print("- ablation_table.tex: LaTeX table for the paper")

if __name__ == '__main__':
    main()