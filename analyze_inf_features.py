#!/usr/bin/env python3
"""
Analyze which features have inf/-inf and what happens during normalization
"""

import numpy as np
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector

def analyze_inf_handling():
    collector = MeteoGaliciaCollector()
    norm_stats_path = collector.data_dir / "normalization_stats.npz"

    if not norm_stats_path.exists():
        print(f"✗ Normalization stats not found")
        return

    norm_stats = np.load(norm_stats_path)
    feature_mins = norm_stats['feature_mins']
    feature_maxs = norm_stats['feature_maxs']

    # Find inf features
    inf_min_indices = np.where(np.isinf(feature_mins))[0]
    inf_max_indices = np.where(np.isinf(feature_maxs))[0]

    print("=" * 70)
    print("ANALYZING INF FEATURES")
    print("=" * 70)

    print(f"\nFeatures with inf in min: {len(inf_min_indices)}")
    print(f"Features with inf in max: {len(inf_max_indices)}")
    print(f"\nIndices: {inf_min_indices.tolist()}")

    # These features had NO valid data points during normalization
    print("\n⚠ Features with inf/-inf had ZERO valid data points!")
    print("   This means they were always either:")
    print("   - Missing (mask = 0)")
    print("   - Invalid markers (-9999, -1000)")

    # What happens during normalization with inf/-inf?
    print("\n" + "=" * 70)
    print("WHAT HAPPENS DURING NORMALIZATION?")
    print("=" * 70)

    # Simulate normalization with inf/-inf
    feat_min = np.inf
    feat_max = -np.inf
    test_value = 15.0  # Some weather value

    print(f"\nIf feat_min = {feat_min}, feat_max = {feat_max}:")
    print(f"  feat_max > feat_min? {feat_max > feat_min}")
    print(f"  → Normalization SKIPPED (condition fails)")
    print(f"  → Value stays as-is: {test_value}")

    # But wait, the value should be -1000 (missing) if there's no data
    print("\n  BUT: If feature had no valid data, it should always be -1000")
    print("  → Then gets replaced with -2 (normalized invalid marker)")
    print("  → Model sees -2 (invalid) consistently")

    # This is probably fine - model learns to ignore -2
    print("\n✓ This is actually OKAY!")
    print("  - Features with inf/-inf have no valid data")
    print("  - Values are always -1000 (missing)")
    print("  - Get normalized to -2")
    print("  - Model learns to ignore -2 (like mask)")

    # So why do predictions suck?
    print("\n" + "=" * 70)
    print("SO WHY ARE PREDICTIONS BAD?")
    print("=" * 70)

    # Find the 71639 outlier
    non_inf_maxs = feature_maxs[~np.isinf(feature_maxs)]
    extreme_idx = np.where(feature_maxs == non_inf_maxs.max())[0]

    print(f"\nThe REAL problem: Feature {extreme_idx[0]}")
    print(f"  Max value: {feature_maxs[extreme_idx[0]]:.2f}")
    print(f"  Min value: {feature_mins[extreme_idx[0]]:.2f}")
    print(f"  Range: {feature_maxs[extreme_idx[0]] - feature_mins[extreme_idx[0]]:.2f}")

    print("\nThis extreme value distorts normalization!")
    print("  15°C normalized with max=71639: (15 - min) / (71639 - min) ≈ -0.999")

    # Show top 10 extreme features
    print("\n" + "=" * 70)
    print("TOP 10 EXTREME FEATURES (excluding inf)")
    print("=" * 70)

    valid_features = ~(np.isinf(feature_mins) | np.isinf(feature_maxs))
    ranges = feature_maxs - feature_mins
    ranges_valid = ranges[valid_features]
    indices_valid = np.where(valid_features)[0]

    # Sort by range
    sorted_idx = np.argsort(ranges_valid)[::-1][:10]

    print(f"\n{'Index':<8} {'Min':<15} {'Max':<15} {'Range':<15}")
    print("-" * 60)
    for i in sorted_idx:
        idx = indices_valid[i]
        print(f"{idx:<8} {feature_mins[idx]:<15.2f} {feature_maxs[idx]:<15.2f} {ranges[idx]:<15.2f}")

if __name__ == "__main__":
    analyze_inf_handling()
