import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_depth_comparison_plot(df, metric_name, output_dir):
    """Create bar plots comparing Mono vs Fused depth metrics"""
    plt.figure(figsize=(10, 6))
    
    # Extract numeric values from string format (removing confidence intervals)
    df = df.copy()
    df[metric_name] = df[metric_name].apply(lambda x: float(str(x).split('±')[0].strip()))
    
    # Create grouped bar plot
    x = np.arange(len(df['Range'].unique()))
    width = 0.35
    
    mono_mask = df['Method'] == 'Mono'
    fused_mask = df['Method'] == 'Fused'
    
    plt.bar(x - width/2, df[mono_mask][metric_name], width, label='Mono', color='lightcoral')
    plt.bar(x + width/2, df[fused_mask][metric_name], width, label='Fused', color='lightblue')
    
    plt.xlabel('Range')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} by Range')
    plt.xticks(x, df['Range'].unique())
    plt.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric_name.lower().replace(" ", "_")}_comparison.png'))
    plt.close()

def create_ttc_distribution_plot(df, output_dir):
    """Create TTC distribution plots"""
    plt.figure(figsize=(10, 6))
    
    valid_ttc = df[np.isfinite(df['ttc'])]['ttc']
    sns.histplot(data=valid_ttc, bins=30)
    plt.xlabel('Time-to-Collision (s)')
    plt.ylabel('Count')
    plt.title('TTC Distribution (Valid Measurements)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ttc_distribution.png'))
    plt.close()

def create_warning_analysis_plot(df, output_dir):
    """Create warning analysis visualization"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data
    warning_data = pd.DataFrame({
        'Type': ['Raw', 'Stable', 'Raw (Fused)', 'Stable (Fused)'],
        'True': [
            df['warn_raw'].sum(),
            df['warn_stable'].sum(),
            df['warn_raw_fused'].sum(),
            df['warn_stable_fused'].sum()
        ],
        'False': [
            len(df) - df['warn_raw'].sum(),
            len(df) - df['warn_stable'].sum(),
            len(df) - df['warn_raw_fused'].sum(),
            len(df) - df['warn_stable_fused'].sum()
        ]
    })
    
    # Create stacked bar plot
    warning_data.plot(x='Type', kind='bar', stacked=True)
    plt.title('Warning Distribution by Type')
    plt.xlabel('Warning Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'warning_distribution.png'))
    plt.close()

def create_summary_dashboard(depth_df, obj_df, output_dir):
    """Create comprehensive summary dashboard"""
    # Create main figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 3, figure=fig)
    
    # 1. Depth Metrics (RMSE)
    ax1 = fig.add_subplot(gs[0, 0:2])
    create_depth_subplot(depth_df, 'RMSE', ax1)
    
    # 2. Warning Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    create_warnings_subplot(obj_df, ax2)
    
    # 3. TTC Distribution
    ax3 = fig.add_subplot(gs[1, 0])
    create_ttc_subplot(obj_df, ax3)
    
    # 4. Class Distribution
    ax4 = fig.add_subplot(gs[1, 1:])
    create_class_subplot(obj_df[obj_df['warn_raw'] | obj_df['warn_raw_fused']], ax4)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), bbox_inches='tight', dpi=300)
    plt.close()

def create_depth_subplot(df, metric_name, ax):
    """Helper function for depth metrics subplot"""
    df = df.copy()
    df[metric_name] = df[metric_name].apply(lambda x: float(str(x).split('±')[0].strip()))
    
    x = np.arange(len(df['Range'].unique()))
    width = 0.35
    
    mono_mask = df['Method'] == 'Mono'
    fused_mask = df['Method'] == 'Fused'
    
    ax.bar(x - width/2, df[mono_mask][metric_name], width, label='Mono', color='lightcoral')
    ax.bar(x + width/2, df[fused_mask][metric_name], width, label='Fused', color='lightblue')
    
    ax.set_xlabel('Range')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} by Range')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Range'].unique(), rotation=45)
    ax.legend()

def create_warnings_subplot(df, ax):
    """Helper function for warnings subplot"""
    warning_counts = [
        df['warn_raw'].sum(),
        df['warn_stable'].sum(),
        df['warn_raw_fused'].sum(),
        df['warn_stable_fused'].sum()
    ]
    labels = ['Raw', 'Stable', 'Raw\n(Fused)', 'Stable\n(Fused)']
    
    ax.bar(labels, warning_counts)
    ax.set_title('Warning Counts')
    ax.set_ylabel('Count')
    plt.setp(ax.get_xticklabels(), rotation=45)

def create_ttc_subplot(df, ax):
    """Helper function for TTC subplot"""
    valid_ttc = df[np.isfinite(df['ttc'])]['ttc']
    sns.histplot(data=valid_ttc, bins=30, ax=ax)
    ax.set_xlabel('Time-to-Collision (s)')
    ax.set_ylabel('Count')
    ax.set_title('TTC Distribution')

def create_class_subplot(df, ax):
    """Helper function for class distribution subplot"""
    class_counts = df['class'].value_counts()
    class_counts.plot(kind='bar', ax=ax)
    ax.set_title('Warning Distribution by Class')
    ax.set_ylabel('Count')
    plt.setp(ax.get_xticklabels(), rotation=45)