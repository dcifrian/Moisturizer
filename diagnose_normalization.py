#!/usr/bin/env python3
"""
Diagnose normalization statistics to find problematic features
"""

import numpy as np
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector

def diagnose_normalization():
    collector = MeteoGaliciaCollector()

    # Check if normalization stats exist
    norm_stats_path = collector.data_dir / "normalization_stats.npz"
    if not norm_stats_path.exists():
        print(f"✗ Normalization stats not found at {norm_stats_path}")
        return

    print("=" * 60)
    print("DIAGNOSING NORMALIZATION STATISTICS")
    print("=" * 60)

    norm_stats = np.load(norm_stats_path)
    feature_mins = norm_stats['feature_mins']
    feature_maxs = norm_stats['feature_maxs']
    target_min = norm_stats['target_min']
    target_max = norm_stats['target_max']

    print(f"\nTarget range: [{target_min:.6f}, {target_max:.6f}]")
    print(f"Number of features: {len(feature_mins)}")

    # Find problematic features
    print("\n" + "=" * 60)
    print("LOOKING FOR EXTREME VALUES")
    print("=" * 60)

    # Check for extreme max values
    extreme_threshold = 1000.0  # Values above this are suspicious
    extreme_features = []

    for feat_idx in range(len(feature_mins)):
        feat_min = feature_mins[feat_idx]
        feat_max = feature_maxs[feat_idx]
        feat_range = feat_max - feat_min

        if abs(feat_max) > extreme_threshold or abs(feat_min) > extreme_threshold or feat_range > extreme_threshold:
            extreme_features.append({
                'idx': feat_idx,
                'min': feat_min,
                'max': feat_max,
                'range': feat_range
            })

    if extreme_features:
        print(f"\n⚠ Found {len(extreme_features)} features with extreme values:")
        print(f"{'Index':<8} {'Min':<15} {'Max':<15} {'Range':<15}")
        print("-" * 60)
        for feat in extreme_features[:20]:  # Show top 20
            print(f"{feat['idx']:<8} {feat['min']:<15.2f} {feat['max']:<15.2f} {feat['range']:<15.2f}")

        if len(extreme_features) > 20:
            print(f"... and {len(extreme_features) - 20} more")
    else:
        print("\n✓ No extreme values found")

    # Check for inf/nan
    print("\n" + "=" * 60)
    print("CHECKING FOR INF/NAN")
    print("=" * 60)

    inf_count = np.isinf(feature_mins).sum() + np.isinf(feature_maxs).sum()
    nan_count = np.isnan(feature_mins).sum() + np.isnan(feature_maxs).sum()

    if inf_count > 0:
        print(f"⚠ Found {inf_count} inf values")
        inf_mins = np.where(np.isinf(feature_mins))[0]
        inf_maxs = np.where(np.isinf(feature_maxs))[0]
        if len(inf_mins) > 0:
            print(f"  Feature mins with inf: {inf_mins[:10]}")
        if len(inf_maxs) > 0:
            print(f"  Feature maxs with inf: {inf_maxs[:10]}")
    else:
        print("✓ No inf values")

    if nan_count > 0:
        print(f"⚠ Found {nan_count} nan values")
    else:
        print("✓ No nan values")

    # Show overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS")
    print("=" * 60)
    print(f"Feature mins: [{feature_mins.min():.2f}, {feature_mins.max():.2f}]")
    print(f"Feature maxs: [{feature_maxs.min():.2f}, {feature_maxs.max():.2f}]")
    print(f"Feature ranges: [{(feature_maxs - feature_mins).min():.2f}, {(feature_maxs - feature_mins).max():.2f}]")

    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if extreme_features:
        print("\n⚠ Your normalization stats contain extreme values!")
        print("\nThis could be causing poor predictions. Options:")
        print("\n1. REGENERATE dataset from scratch (RECOMMENDED):")
        print("   - Clean slate, guaranteed correctness")
        print("   - Takes ~24 hours")
        print("   - Run: python test_performance.py (or your dataset generation script)")
        print("\n2. Investigate and patch:")
        print("   - Check which parameters have extreme values")
        print("   - May need to check raw data quality")
        print("   - Could be legitimate data (e.g., cumulative precipitation)")
    else:
        print("\n✓ Normalization stats look reasonable!")

if __name__ == "__main__":
    diagnose_normalization()
