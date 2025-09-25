import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_and_visualize_metrics(data_dir):
    """Save metrics to CSV and create visualizations in Eval/output"""
    # Set output directory
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(eval_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving outputs to: {output_dir}")
    
    # Load and process metrics
    obj_df = pd.read_csv(os.path.join(data_dir, 'object_depth_metrics.csv'))
    
    # Save metrics to CSV
    obj_metrics_path = os.path.join(output_dir, 'object_metrics.csv')
    obj_df.to_csv(obj_metrics_path, index=False)
    print(f"Saved object metrics to: {obj_metrics_path}")
    
    # Create visualizations
    create_warning_plot(obj_df, output_dir)
    create_ttc_plot(obj_df, output_dir)
    create_depth_distribution_plot(obj_df, output_dir)
    
    print("Created visualization plots successfully")

def create_warning_plot(df, output_dir):
    plt.figure(figsize=(10, 6))
    warning_data = pd.DataFrame({
        'Warning Type': ['Raw', 'Stable', 'Raw (Fused)', 'Stable (Fused)'],
        'True Count': [
            df['warn_raw'].sum(),
            df['warn_stable'].sum(),
            df['warn_raw_fused'].sum(),
            df['warn_stable_fused'].sum()
        ]
    })
    sns.barplot(data=warning_data, x='Warning Type', y='True Count')
    plt.title('Warning Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'warning_distribution.png'))
    plt.close()

def create_ttc_plot(df, output_dir):
    plt.figure(figsize=(10, 6))
    valid_ttc = df[np.isfinite(df['ttc'])]['ttc']
    sns.histplot(data=valid_ttc, bins=30)
    plt.title('TTC Distribution (Valid Values Only)')
    plt.xlabel('Time to Collision (s)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttc_distribution.png'))
    plt.close()

def create_depth_distribution_plot(df, output_dir):
    plt.figure(figsize=(12, 6))
    warning_cases = df[df['warn_raw'] | df['warn_raw_fused']]
    
    data = []
    for method in ['mono_median_depth', 'lidar_median_depth', 'fused_median_depth']:
        depth_values = warning_cases[method].dropna()
        data.extend([(d, method.split('_')[0].title()) for d in depth_values])
    
    plot_df = pd.DataFrame(data, columns=['Depth', 'Method'])
    sns.boxplot(data=plot_df, x='Method', y='Depth')
    plt.title('Depth Distribution for Warning Cases')
    plt.ylabel('Depth (m)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'depth_distribution.png'))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fused_output',
                       help='Directory containing evaluation results')
    args = parser.parse_args()
    save_and_visualize_metrics(args.data_dir)