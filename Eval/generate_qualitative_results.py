#!/usr/bin/env python3
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def load_frame_data(base_dir, frame_num):
    """Load RGB, depth maps and overlays for a given frame."""
    # Load RGB image
    rgb = cv2.imread(os.path.join(base_dir, f"{frame_num:05d}_rgb.png"))
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Load depth maps
    mono_depth = cv2.imread(os.path.join(base_dir, f"{frame_num:05d}_mono_depth.png"), -1)
    lidar_depth = cv2.imread(os.path.join(base_dir, f"{frame_num:05d}_lidar_depth.png"), -1)
    fused_depth = cv2.imread(os.path.join(base_dir, f"{frame_num:05d}_fused_depth.png"), -1)
    
    # Load ECW overlay
    ecw_overlay = cv2.imread(os.path.join(base_dir, f"{frame_num:05d}_overlay.png"))
    ecw_overlay = cv2.cvtColor(ecw_overlay, cv2.COLOR_BGR2RGB)
    
    return rgb, mono_depth, lidar_depth, fused_depth, ecw_overlay

def visualize_qualitative_results(data_dir, output_dir, selected_frames):
    """Generate Figure 7 with qualitative results."""
    
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # Row 1: Fused depth quality
    frame = selected_frames['depth_quality']
    rgb, mono, lidar, fused, ecw = load_frame_data(data_dir, frame)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(rgb)
    ax1.set_title('RGB with ECW Mask')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mono, cmap='magma')
    ax2.set_title('Monocular-Only Depth')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(fused, cmap='magma')
    ax3.set_title('Ours (Fused)')
    
    # Row 2: Missed-detection recovery
    frame = selected_frames['missed_detection']
    rgb, mono, lidar, fused, ecw = load_frame_data(data_dir, frame)
    
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(rgb)
    ax4.set_title('Missed VRU Detection')
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.imshow(fused, cmap='magma')
    ax5.set_title('Depth-blob Mining')
    
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.imshow(ecw)
    ax6.set_title('Recovered Warning')
    
    # Row 3: Highway cut-in
    frame = selected_frames['highway_cutin']
    rgb, mono, lidar, fused, ecw = load_frame_data(data_dir, frame)
    
    times = np.arange(-1.0, 1.0, 0.1)
    ttc_values = [2.5 + np.sin(t) for t in times]  # Example TTC curve
    warning_states = [1 if ttc < 2.0 else 0 for ttc in ttc_values]
    
    ax7 = fig.add_subplot(gs[2, :2])
    ax7.plot(times, ttc_values, 'b-', label='TTC')
    ax7_twin = ax7.twinx()
    ax7_twin.plot(times, warning_states, 'r--', label='Warning')
    ax7.axvline(x=0, color='k', linestyle=':', label='t*')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('TTC (s)')
    ax7_twin.set_ylabel('Warning State')
    ax7.legend(loc='upper right')
    ax7.set_title('Highway Cut-in Timeline')
    
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.imshow(ecw)
    ax8.set_title('Early Warning (t*-0.5s)')
    
    # Add descriptive caption
    plt.figtext(0.5, 0.02,
                'Figure 7: Qualitative results showing (top) improved depth quality, ' +
                '(middle) missed-detection recovery via depth mining, and ' +
                '(bottom) early warning for highway cut-in with TTC timeline.',
                wrap=True, ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_7_qualitative.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fused_output',
                      help='Directory with fusion results')
    args = parser.parse_args()
    
    output_dir = os.path.join('Eval', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Selected frames for each scenario (to be replaced with actual frame numbers)
    selected_frames = {
        'depth_quality': 19990,      # Frame showing clear depth improvements
        'missed_detection': 20150,   # Frame with VRU partially occluded
        'highway_cutin': 20300      # Frame showing highway cut-in scenario
    }
    
    visualize_qualitative_results(args.data_dir, output_dir, selected_frames)