#!/usr/bin/env python3
"""
Test script to analyze normalization outliers in the AugmentedLiveDataset.

This script:
1. Loads the dataset (with sampling-based normalization stats)
2. Iterates through samples and checks for values outside the expected [-1, 1] range
3. Reports statistics per feature: count, average deviation, worst outlier

Usage:
    # Quick test with tiny dataset (4 days, seq_length 2):
    python test_normalization_outliers.py --tiny

    # Full analysis (10+ years of data):
    python test_normalization_outliers.py

    # Analyze a subset of samples:
    python test_normalization_outliers.py --max-samples 100000
"""

import argparse
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm
import sys


def analyze_outliers(dataset, max_samples=None, batch_size=1000):
    """
    Analyze normalization outliers across the dataset.

    Returns a dict with statistics per feature index.
    """
    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    # Get feature dimension from first sample
    sample = dataset[0]
    features = sample['features'].numpy()  # [seq_length, n_features]
    seq_length, n_features = features.shape

    print(f"\nAnalyzing {n_samples:,} samples...")
    print(f"  Feature shape per sample: [{seq_length}, {n_features}]")
    print(f"  Total feature values to check: {n_samples * seq_length * n_features:,}")

    # Statistics per feature
    # For each feature, track:
    # - count of values outside [-1, 1] (mild outliers)
    # - count of values outside [-2, 2] (severe outliers)
    # - sum of deviations (for computing average)
    # - worst outlier (most extreme value)

    mild_outlier_counts = np.zeros(n_features, dtype=np.int64)
    severe_outlier_counts = np.zeros(n_features, dtype=np.int64)
    deviation_sums = np.zeros(n_features, dtype=np.float64)
    worst_values = np.zeros(n_features, dtype=np.float32)  # Most extreme normalized value
    total_values = np.zeros(n_features, dtype=np.int64)

    # Also track the invalid marker (-2) to distinguish from outliers
    invalid_counts = np.zeros(n_features, dtype=np.int64)

    # Process samples
    for idx in tqdm(range(n_samples), desc="Analyzing", unit="sample"):
        sample = dataset[idx]
        features = sample['features'].numpy()  # [seq_length, n_features]

        for feat_idx in range(n_features):
            feat_values = features[:, feat_idx]

            # Count invalid markers (exactly -2.0)
            invalid_mask = np.isclose(feat_values, -2.0, atol=1e-6)
            invalid_counts[feat_idx] += invalid_mask.sum()

            # Exclude invalid markers for outlier analysis
            valid_values = feat_values[~invalid_mask]
            total_values[feat_idx] += len(valid_values)

            if len(valid_values) == 0:
                continue

            # Mild outliers: outside [-1, 1]
            mild_mask = (valid_values < -1.0) | (valid_values > 1.0)
            mild_count = mild_mask.sum()
            mild_outlier_counts[feat_idx] += mild_count

            # Severe outliers: outside [-2, 2]
            severe_mask = (valid_values < -2.0) | (valid_values > 2.0)
            severe_count = severe_mask.sum()
            severe_outlier_counts[feat_idx] += severe_count

            # Track deviations (distance from [-1, 1] range)
            if mild_count > 0:
                outlier_values = valid_values[mild_mask]
                # Deviation = how far outside [-1, 1]
                deviations = np.maximum(np.abs(outlier_values) - 1.0, 0)
                deviation_sums[feat_idx] += deviations.sum()

            # Track worst (most extreme) value
            if len(valid_values) > 0:
                max_abs = np.max(np.abs(valid_values))
                if max_abs > np.abs(worst_values[feat_idx]):
                    # Store the actual value (with sign) not just abs
                    extreme_idx = np.argmax(np.abs(valid_values))
                    worst_values[feat_idx] = valid_values[extreme_idx]

    return {
        'n_features': n_features,
        'n_samples': n_samples,
        'seq_length': seq_length,
        'mild_outlier_counts': mild_outlier_counts,
        'severe_outlier_counts': severe_outlier_counts,
        'deviation_sums': deviation_sums,
        'worst_values': worst_values,
        'total_values': total_values,
        'invalid_counts': invalid_counts,
    }


def print_report(stats, feature_names=None):
    """Print a detailed report of outlier statistics."""
    n_features = stats['n_features']
    n_samples = stats['n_samples']
    seq_length = stats['seq_length']

    mild_counts = stats['mild_outlier_counts']
    severe_counts = stats['severe_outlier_counts']
    deviation_sums = stats['deviation_sums']
    worst_values = stats['worst_values']
    total_values = stats['total_values']
    invalid_counts = stats['invalid_counts']

    print("\n" + "=" * 100)
    print("NORMALIZATION OUTLIER ANALYSIS REPORT")
    print("=" * 100)
    print(f"\nDataset: {n_samples:,} samples × {seq_length} timesteps × {n_features} features")
    print(f"Total values analyzed: {total_values.sum():,}")
    print(f"Total invalid markers (-2.0): {invalid_counts.sum():,}")

    # Summary statistics
    total_mild = mild_counts.sum()
    total_severe = severe_counts.sum()
    features_with_mild = (mild_counts > 0).sum()
    features_with_severe = (severe_counts > 0).sum()

    print(f"\n--- SUMMARY ---")
    print(f"Features with mild outliers (outside [-1,1]):   {features_with_mild}/{n_features}")
    print(f"Features with severe outliers (outside [-2,2]): {features_with_severe}/{n_features}")
    print(f"Total mild outliers:   {total_mild:,} ({100*total_mild/total_values.sum():.4f}%)")
    print(f"Total severe outliers: {total_severe:,} ({100*total_severe/total_values.sum():.6f}%)")

    # Per-feature report (only features with outliers)
    print(f"\n--- PER-FEATURE DETAILS (features with outliers only) ---")
    print(f"{'Idx':>5} {'Name':>30} {'Mild':>12} {'Severe':>12} {'Avg Dev':>10} {'Worst':>12} {'Invalid%':>10}")
    print("-" * 100)

    # Sort by severe outliers first, then mild
    feature_order = np.lexsort((mild_counts, severe_counts))[::-1]

    shown = 0
    for feat_idx in feature_order:
        if mild_counts[feat_idx] == 0 and severe_counts[feat_idx] == 0:
            continue

        name = feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"feat_{feat_idx}"
        if len(name) > 30:
            name = name[:27] + "..."

        avg_dev = deviation_sums[feat_idx] / mild_counts[feat_idx] if mild_counts[feat_idx] > 0 else 0
        invalid_pct = 100 * invalid_counts[feat_idx] / (total_values[feat_idx] + invalid_counts[feat_idx]) if total_values[feat_idx] + invalid_counts[feat_idx] > 0 else 0

        print(f"{feat_idx:>5} {name:>30} {mild_counts[feat_idx]:>12,} {severe_counts[feat_idx]:>12,} {avg_dev:>10.4f} {worst_values[feat_idx]:>12.4f} {invalid_pct:>9.2f}%")
        shown += 1

        if shown >= 50:  # Limit output
            remaining = (mild_counts > 0).sum() + (severe_counts > 0).sum() - shown
            if remaining > 0:
                print(f"  ... and {remaining} more features with outliers")
            break

    if shown == 0:
        print("  No outliers detected! All values within [-1, 1] range.")

    # Top 10 worst outliers
    print(f"\n--- TOP 10 MOST EXTREME VALUES ---")
    worst_order = np.argsort(np.abs(worst_values))[::-1][:10]
    for rank, feat_idx in enumerate(worst_order, 1):
        if np.abs(worst_values[feat_idx]) <= 1.0:
            break
        name = feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"feat_{feat_idx}"
        print(f"  {rank:>2}. Feature {feat_idx:>3} ({name[:30]:>30}): {worst_values[feat_idx]:>12.4f}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Analyze normalization outliers in AugmentedLiveDataset')
    parser.add_argument('--tiny', action='store_true',
                        help='Use tiny dataset (4 days, seq_length 2) for quick testing')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to analyze (default: all)')
    parser.add_argument('--save-stats', type=str, default=None,
                        help='Save statistics to .npz file')
    args = parser.parse_args()

    # Import here to avoid import errors during --help
    from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
    from augmented_live import AugmentedLiveDataset

    collector = MeteoGaliciaCollector()

    if args.tiny:
        print("\n" + "=" * 60)
        print("TINY DATASET MODE (4 days, seq_length 2)")
        print("=" * 60)

        # Create a tiny dataset for testing
        _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

        # Use the base SoilMoistureSequenceDataset with minimal settings
        # We can't easily limit days in the live augmented version, so we'll
        # just use a very small max_samples
        print("\nBuilding tiny dataset...")
        dataset = AugmentedLiveDataset.from_base_dataset(
            timeseries=str(collector.timeseries_file),
            stations=str(collector.stations_file),
            nearest=str(collector.nearest_file),
            dense_array_path=str(collector.data_dir / "dense_features.npz"),
            feature_params=filtered_params,
            seq_length=2,  # Minimal sequence length
            n_nearby_available=5,
            n_nearby_in_features=4,
            normalize=True,
        )

        # For tiny test, just check a few samples
        max_samples = args.max_samples if args.max_samples else min(1000, len(dataset))
        feature_names = filtered_params if filtered_params else None

    else:
        print("\n" + "=" * 60)
        print("FULL DATASET ANALYSIS")
        print("=" * 60)

        # Load full dataset
        _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

        print("\nLoading full dataset...")
        dataset = AugmentedLiveDataset.from_base_dataset(
            timeseries=str(collector.timeseries_file),
            stations=str(collector.stations_file),
            nearest=str(collector.nearest_file),
            dense_array_path=str(collector.data_dir / "dense_features.npz"),
            feature_params=filtered_params,
            seq_length=64,
            n_nearby_available=5,
            n_nearby_in_features=4,
            normalize=True,
        )

        max_samples = args.max_samples
        feature_names = filtered_params if filtered_params else None

    # Build feature names including nearby stations
    if feature_names:
        full_feature_names = list(feature_names)  # Target station features
        for i in range(4):  # 4 nearby stations
            full_feature_names.append(f"nearby{i+1}_distance")
            for param in feature_names:
                full_feature_names.append(f"nearby{i+1}_{param}")
            full_feature_names.append(f"nearby{i+1}_soil_moisture")
    else:
        full_feature_names = None

    # Run analysis
    stats = analyze_outliers(dataset, max_samples=max_samples)

    # Print report
    print_report(stats, full_feature_names)

    # Save stats if requested
    if args.save_stats:
        np.savez(args.save_stats, **stats)
        print(f"\nStatistics saved to {args.save_stats}")


if __name__ == "__main__":
    main()
