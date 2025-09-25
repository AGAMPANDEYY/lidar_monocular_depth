import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_depth_plots(depth_df, output_dir):
    """Create depth evaluation plots"""
    # Convert data format
    plot_data = []
    for _, row in depth_df.iterrows():
        method = row['Method']
        range_val = row['Range']
        rmse = float(str(row['RMSE']).split('±')[0])
        absrel = float(str(row['AbsRel']).split('±')[0])
        plot_data.append({'Method': method, 'Range': range_val, 'Metric': 'RMSE', 'Value': rmse})
        plot_data.append({'Method': method, 'Range': range_val, 'Metric': 'AbsRel', 'Value': absrel})
    
    plot_df = pd.DataFrame(plot_data)
    
    # Create separate plots for RMSE and AbsRel
    metrics = ['RMSE', 'AbsRel']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        metric_data = plot_df[plot_df['Metric'] == metric]
        sns.barplot(data=metric_data, x='Range', y='Value', hue='Method')
        plt.title(f'{metric} by Range')
        plt.ylabel(f'{metric} Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric.lower()}_by_range.png'))
        plt.close()

def process_ecw_metrics(obj_df):
    """Process ECW metrics properly"""
    ecw_metrics = {
        'total_detections': len(obj_df),
        'warning_stats': {
            'raw': obj_df['warn_raw'].value_counts().to_dict(),
            'stable': obj_df['warn_stable'].value_counts().to_dict(),
            'raw_fused': obj_df['warn_raw_fused'].value_counts().to_dict(),
            'stable_fused': obj_df['warn_stable_fused'].value_counts().to_dict()
        },
        'class_distribution': obj_df[obj_df['warn_raw'] | obj_df['warn_raw_fused']]['class'].value_counts().to_dict(),
        'ttc_stats': {
            'valid_count': np.isfinite(obj_df['ttc']).sum(),
            'inf_count': np.isinf(obj_df['ttc']).sum(),
            'nan_count': np.isnan(obj_df['ttc']).sum(),
            'statistics': obj_df[np.isfinite(obj_df['ttc'])]['ttc'].describe().to_dict()
        }
    }
    return ecw_metrics