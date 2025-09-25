#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

def compute_event_metrics(gt_events, pred_events, tolerance_s=0.5, fps=25.0):
    """
    Compute event-level metrics for ECW system
    
    Args:
        gt_events: List of (start_frame, end_frame) tuples for ground truth events
        pred_events: List of (start_frame, end_frame) tuples for predicted events
        tolerance_s: Time tolerance in seconds for matching events
        fps: Frames per second of the video
    
    Returns:
        dict with metrics including precision, recall, F1, and average lead time
    """
    tol_frames = int(tolerance_s * fps)
    matched_gt = set()
    matched_pred = set()
    lead_times = []
    
    # Find matches within tolerance
    for i, (gt_start, gt_end) in enumerate(gt_events):
        for j, (pred_start, pred_end) in enumerate(pred_events):
            # Check if events overlap or are within tolerance
            if (gt_start - tol_frames <= pred_end) and (gt_end + tol_frames >= pred_start):
                # Event is matched
                matched_gt.add(i)
                matched_pred.add(j)
                # Compute lead time (positive means early warning)
                lead_time = (gt_start - pred_start) / fps
                lead_times.append(lead_time)
                break
    
    tp = len(matched_gt)  # Each matched pair counts as one true positive
    fp = len(pred_events) - len(matched_pred)  # Unmatched predictions
    fn = len(gt_events) - len(matched_gt)  # Unmatched ground truth
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'lead_times': lead_times,
        'avg_lead_time': np.mean(lead_times) if lead_times else 0.0
    }
    
    return metrics

def find_events(warning_series, min_frames=3):
    """
    Convert a binary warning signal into a list of events
    
    Args:
        warning_series: pandas Series with boolean warning values
        min_frames: Minimum event duration in frames
    
    Returns:
        List of (start_frame, end_frame) tuples
    """
    events = []
    in_event = False
    start_frame = None
    
    for frame, warning in warning_series.items():
        if warning and not in_event:
            # Start of new event
            start_frame = frame
            in_event = True
        elif not warning and in_event:
            # End of current event
            if frame - start_frame >= min_frames:
                events.append((start_frame, frame - 1))
            in_event = False
    
    # Handle event in progress at end of sequence
    if in_event and (warning_series.index[-1] - start_frame >= min_frames):
        events.append((start_frame, warning_series.index[-1]))
    
    return events

def compute_confidence_intervals(metrics, n_bootstrap=1000, confidence_level=0.95):
    """
    Compute confidence intervals for ECW metrics using bootstrapping
    
    Args:
        metrics: Dict with arrays/lists of per-sequence metrics
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
    
    Returns:
        Dict with mean and confidence intervals for each metric
    """
    results = {}
    for metric_name, values in metrics.items():
        if isinstance(values, (list, np.ndarray)) and len(values) > 1:
            values = np.array(values)
            bs_stats = bootstrap((values,), np.mean, n_resamples=n_bootstrap, 
                              confidence_level=confidence_level)
            results[metric_name] = {
                'mean': np.mean(values),
                'ci_lower': bs_stats.confidence_interval.low,
                'ci_upper': bs_stats.confidence_interval.high
            }
        else:
            results[metric_name] = {
                'mean': np.mean(values) if len(values) > 0 else 0.0,
                'ci_lower': None,
                'ci_upper': None
            }
            
    return results

def compute_sequence_metrics(obj_df, fps=25.0, min_frames=3, tolerance_s=0.5):
    """
    Compute ECW metrics for a sequence of frames
    
    Args:
        obj_df: DataFrame with object detections and warnings
        fps: Frames per second
        min_frames: Minimum event duration in frames
        tolerance_s: Time tolerance for event matching in seconds
        
    Returns:
        Dict with computed metrics
    """
    # Convert warnings to events
    mono_events = find_events(obj_df['warn_stable_mono'].fillna(False), min_frames)
    fused_events = find_events(obj_df['warn_stable_fused'].fillna(False), min_frames)
    gt_events = find_events(obj_df['gt_hazard'].fillna(False), min_frames)
    
    # Compute metrics for both methods
    mono_metrics = compute_event_metrics(gt_events, mono_events, tolerance_s, fps)
    fused_metrics = compute_event_metrics(gt_events, fused_events, tolerance_s, fps)
    
    return {
        'mono': mono_metrics,
        'fused': fused_metrics,
        'n_gt_events': len(gt_events),
        'n_mono_events': len(mono_events),
        'n_fused_events': len(fused_events)
    }
