#!/usr/bin/env python3
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_visualization(frame_data, output_path, scenario_type):
    """Create qualitative visualization for a specific scenario."""
    plt.style.use('seaborn-v0_8-paper')  # Updated style name for newer versions
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

    fig = plt.figure(figsize=(15, 10))
    
    # RGB with boxes and ECW
    ax1 = plt.subplot(221)
    ax1.imshow(frame_data['rgb'])
    if 'boxes' in frame_data:
        for box in frame_data['boxes']:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                  linewidth=2, edgecolor='r', facecolor='none')
            ax1.add_patch(rect)
    ax1.set_title('RGB with Detection & Warnings')
    
    # Monocular depth
    ax2 = plt.subplot(222)
    mono_depth = ax2.imshow(frame_data['mono_depth'], cmap='magma')
    plt.colorbar(mono_depth, ax=ax2)
    ax2.set_title('Monocular-Only Depth')
    
    # LiDAR overlay
    ax3 = plt.subplot(223)
    ax3.imshow(frame_data['rgb'])  # Base RGB image
    if 'lidar_points' in frame_data:
        lidar_scatter = ax3.scatter(frame_data['lidar_points'][:, 0], 
                                frame_data['lidar_points'][:, 1],
                                c=frame_data['lidar_points'][:, 2],
                                cmap='magma', alpha=0.5)
        plt.colorbar(lidar_scatter, ax=ax3)
    ax3.set_title('LiDAR Points Overlay')
    
    # Fused depth
    ax4 = plt.subplot(224)
    fused_depth = ax4.imshow(frame_data['fused_depth'], cmap='magma')
    plt.colorbar(fused_depth, ax=ax4)
    ax4.set_title('Fused Depth')
    
    plt.suptitle(f'Scenario: {scenario_type}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_timeline_strip(data, output_path):
    """Create timeline visualization showing TTC, warnings, and trigger moment."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 10,
        'legend.fontsize': 8
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), height_ratios=[2, 1])
    
    # TTC curve
    ax1.plot(data['time'], data['ttc'], 'b-', label='TTC')
    ax1.axhline(y=data['ttc_threshold'], color='r', linestyle='--', label='Warning Threshold')
    if 't_star' in data:
        ax1.axvline(x=data['t_star'], color='g', linestyle=':', label='t*')
    ax1.set_ylabel('TTC (s)')
    ax1.grid(True)
    ax1.legend()
    
    # Warning state
    ax2.fill_between(data['time'], 0, data['warning_state'], 
                     color='r', alpha=0.3, label='Warning Active')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Warning\nState')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Off', 'On'])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_qualitative_results(data_dir):
    """Generate all qualitative visualizations."""
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure matplotlib is properly configured
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
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
            # Load data for the frame
            base_path = os.path.join(data_dir, "debug", frame_id)
            
            # Load RGB image
            try:
                rgb_path = f"{base_path}_rgb.png"
                if os.path.exists(rgb_path):
                    rgb = plt.imread(rgb_path)
                else:
                    print(f"Warning: RGB image not found at {rgb_path}")
                    continue
            except Exception as e:
                print(f"Error loading RGB image: {str(e)}")
                continue
                
            # Load numpy arrays
            try:
                frame_data = {
                    'rgb': rgb,
                    'mono_depth': np.load(f"{base_path}_mono_depth.npy"),
                    'fused_depth': np.load(f"{base_path}_fused_depth.npy"),
                    'lidar_depth': np.load(f"{base_path}_lidar_depth.npy"),
                    'lidar_mask': np.load(f"{base_path}_lidar_mask.npy")
                }
                
                # Try to load LiDAR projected points
                try:
                    frame_data['lidar_points'] = np.load(f"{base_path}_lidar_projected.npy")
                except FileNotFoundError:
                    print(f"Warning: LiDAR projected points not found for {frame_id}")
                    
            except Exception as e:
                print(f"Error loading numpy arrays: {str(e)}")
                continue
            
            # Create visualization
            output_path = os.path.join(output_dir, f"{scenario_type}.png")
            create_visualization(frame_data, output_path, scenario_type)
            print(f"Saved visualization to {output_path}")
            
        except Exception as e:
            print(f"Failed to generate {scenario_type} visualization: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Set matplotlib backend first
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fused_output',
                       help='Directory containing evaluation results')
    args = parser.parse_args()
    generate_qualitative_results(args.data_dir)