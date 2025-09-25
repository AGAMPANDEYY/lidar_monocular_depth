#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from visualization_utils import (
    create_depth_comparison_plot,
    create_ttc_distribution_plot,
    create_warning_analysis_plot,
    create_summary_dashboard
)

def evaluate_and_visualize(data_dir):
    """Main function to evaluate results and create visualizations"""
    # Create output directory in Eval/output
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nVisualization outputs will be saved to: {output_dir}")
    
    # Load data
    obj_df = pd.read_csv(os.path.join(data_dir, 'object_depth_metrics.csv'))
    dense_results_file = os.path.join(data_dir, 'dense_depth_metrics.csv')
    if os.path.exists(dense_results_file):
        dense_results_df = pd.read_csv(dense_results_file)
    else:
        print("Warning: dense_depth_metrics.csv not found")
        return
    
    # Generate visualizations
    try:
        print("\nGenerating visualization plots...")
        
        # Depth comparison plots
        create_depth_comparison_plot(dense_results_df, 'RMSE', output_dir)
        create_depth_comparison_plot(dense_results_df, 'AbsRel', output_dir)
        
        # TTC and warning distribution plots
        create_ttc_distribution_plot(obj_df, output_dir)
        create_warning_analysis_plot(obj_df, output_dir)
        
        # Comprehensive dashboard
        create_summary_dashboard(dense_results_df, obj_df, output_dir)
        
        print(f"Created visualization plots in: {output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fused_output',
                       help='Directory containing evaluation results')
    args = parser.parse_args()
    evaluate_and_visualize(args.data_dir)