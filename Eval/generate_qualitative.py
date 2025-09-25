#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
import numpy as np
import os
def setup_plot_style():
    """Set up matplotlib style for publication quality plots"""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True
    })

def generate_qualitative_visualizations(data_dir):
    """Generate all qualitative visualizations."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Scenario frames
    scenarios = {
        'fused_depth': '18050',
        'missed_detection': '18075',
        'glare_robustness': '18100',
        'highway_cutin': '18150'
    }
    
    print("\n=== Qualitative Analysis ===")
    
    for scenario_type, frame_id in scenarios.items():
        print(f"Generating visualization for {scenario_type}...")
        try:
            # Load frame data
            base_path = os.path.join(data_dir, "debug", frame_id)
            
            # Load all required data
            frame_data = {
                'rgb': plt.imread(f"{base_path}_rgb.png"),
                'mono_depth': np.load(f"{base_path}_mono_depth.npy"),
                'fused_depth': np.load(f"{base_path}_fused_depth.npy"),
                'lidar_depth': np.load(f"{base_path}_lidar_depth.npy"),
                'lidar_mask': np.load(f"{base_path}_lidar_mask.npy")
            }
            
            # Try to load LiDAR projected points
            try:
                frame_data['lidar_points'] = np.load(f"{base_path}_lidar_projected.npy")
            except FileNotFoundError:
                print(f"Warning: LiDAR projected points not found for frame {frame_id}")
            
            # Create visualization
            fig = plt.figure(figsize=(15, 10))
            setup_plot_style()
            
            # RGB image
            ax1 = plt.subplot(221)
            ax1.imshow(frame_data['rgb'])
            ax1.set_title('RGB Image')
            
            # Monocular depth
            ax2 = plt.subplot(222)
            mono_depth = ax2.imshow(frame_data['mono_depth'], cmap='magma')
            plt.colorbar(mono_depth, ax=ax2)
            ax2.set_title('Monocular-Only Depth')
            
            # LiDAR overlay
            ax3 = plt.subplot(223)
            ax3.imshow(frame_data['rgb'])
            if 'lidar_points' in frame_data:
                points = frame_data['lidar_points']
                if len(points.shape) == 3:  # If points are in image format
                    ax3.imshow(points, cmap='magma', alpha=0.5)
                else:  # If points are x,y,z coordinates
                    scatter = ax3.scatter(points[:, 0], points[:, 1], 
                                       c=points[:, 2], cmap='magma', alpha=0.5)
                    plt.colorbar(scatter, ax=ax3)
            ax3.set_title('LiDAR Points Overlay')
            
            # Fused depth
            ax4 = plt.subplot(224)
            fused_depth = ax4.imshow(frame_data['fused_depth'], cmap='magma')
            plt.colorbar(fused_depth, ax=ax4)
            ax4.set_title('Fused Depth')
            
            plt.suptitle(f'Scenario: {scenario_type}')
            plt.tight_layout()
            
            # Save figure
            output_path = os.path.join(output_dir, f"{scenario_type}.png")
            plt.savefig(output_path)
            plt.close(fig)
            
            print(f"Saved visualization to {output_path}")
            
        except Exception as e:
            print(f"Failed to generate {scenario_type} visualization: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\nQualitative visualizations saved in output directory")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fused_output',
                       help='Directory containing evaluation results')
    args = parser.parse_args()
    generate_qualitative_visualizations(args.data_dir)