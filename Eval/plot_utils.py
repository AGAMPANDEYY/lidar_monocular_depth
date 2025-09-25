import matplotlib.pyplot as plt
import numpy as np

def create_comparison_dashboard(depth_df, output_path):
    """Create a comparison dashboard with proper data formatting"""
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    plt.suptitle('Depth and ECW Performance Summary', fontsize=14)
    
    # Separate Mono and Fused data
    ranges = depth_df['Range'].unique()
    x = np.arange(len(ranges))
    width = 0.35
    
    mono_mask = depth_df['Method'] == 'Mono'
    fused_mask = depth_df['Method'] == 'Fused'
    
    # Plot 1: RMSE by range
    rmse_mono = depth_df[mono_mask]['RMSE'].values
    rmse_fused = depth_df[fused_mask]['RMSE'].values
    
    axes[0,0].bar(x - width/2, rmse_mono, width, label='Mono', color='lightcoral')
    axes[0,0].bar(x + width/2, rmse_fused, width, label='Fused', color='lightblue')
    axes[0,0].set_title('Depth RMSE by Range')
    axes[0,0].set_xticks(x)
    axes[0,0].set_xticklabels(ranges)
    axes[0,0].set_ylabel('RMSE (meters)')
    axes[0,0].legend()
    
    # Plot 2: AbsRel by range
    absrel_mono = depth_df[mono_mask]['AbsRel'].str.split('±').str[0].astype(float).values
    absrel_fused = depth_df[fused_mask]['AbsRel'].str.split('±').str[0].astype(float).values
    
    axes[0,1].bar(x - width/2, absrel_mono, width, label='Mono', color='lightcoral')
    axes[0,1].bar(x + width/2, absrel_fused, width, label='Fused', color='lightblue')
    axes[0,1].set_title('Absolute Relative Error by Range')
    axes[0,1].set_xticks(x)
    axes[0,1].set_xticklabels(ranges)
    axes[0,1].set_ylabel('Absolute Relative Error')
    axes[0,1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()