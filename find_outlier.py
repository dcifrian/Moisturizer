#!/usr/bin/env python3
"""
Find ALL outliers in normalization stats and identify which features they correspond to
"""

import numpy as np
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector
import pandas as pd

def find_all_outliers():
    collector = MeteoGaliciaCollector()

    # Load normalization stats
    norm_stats_path = collector.data_dir / "normalization_stats.npz"
    if not norm_stats_path.exists():
        print(f"✗ No normalization stats found at {norm_stats_path}")
        return

    norm_stats = np.load(norm_stats_path)
    feature_mins = norm_stats['feature_mins']
    feature_maxs = norm_stats['feature_maxs']

    print("=" * 70)
    print("ANALYZING ALL FEATURES IN NORMALIZATION STATS")
    print("=" * 70)
    print(f"\nTotal features in normalization stats: {len(feature_mins)}")

    # Load dense arrays to understand feature structure
    dense_path = collector.data_dir / "dense_features.npz"
    timeseries_df = None

    if dense_path.exists():
        dense_data = np.load(dense_path)
        feature_params = dense_data['feature_params'].tolist()
        print(f"Parameters in dense array: {len(feature_params)}")
        print(f"  Parameters: {feature_params[:5]}... (+{len(feature_params)-5} more)")

        # Load timeseries for checking actual data
        timeseries_df = pd.read_csv(collector.timeseries_file)

    # Figure out feature structure:
    # Final tensor = [target features] + [nearby1: distance + features + soil] + [nearby2...] + [nearby4...]
    # So feature index 110 needs to be mapped back to what it represents

    print("\n" + "=" * 70)
    print("FEATURES WITH SUSPICIOUS RANGES")
    print("=" * 70)
    print(f"\n{'Idx':<5} {'Min':<12} {'Max':<12} {'Range':<12} {'Issue'}")
    print("-" * 70)

    # Known large values that are OK
    utm_params = ['umtx', 'umty']  # UTM coordinates in meters

    outliers = []

    for feat_idx in range(len(feature_mins)):
        feat_min = feature_mins[feat_idx]
        feat_max = feature_maxs[feat_idx]
        feat_range = feat_max - feat_min

        # Skip inf/-inf features (no data)
        if np.isinf(feat_min) or np.isinf(feat_max):
            continue

        issues = []

        # Check for extremely large max values (excluding UTM coords)
        if feat_max > 10000:
            issues.append(f"Max too large: {feat_max:.0f}")

        # Check for extremely large range
        if feat_range > 10000:
            issues.append(f"Range too large: {feat_range:.0f}")

        # Check for unexpected negative values (most weather params shouldn't be very negative)
        if feat_min < -100:
            issues.append(f"Min too negative: {feat_min:.0f}")

        if issues:
            outliers.append({
                'idx': feat_idx,
                'min': feat_min,
                'max': feat_max,
                'range': feat_range,
                'issues': issues
            })
            issue_str = "; ".join(issues)
            print(f"{feat_idx:<5} {feat_min:<12.2f} {feat_max:<12.2f} {feat_range:<12.2f} {issue_str}")

    print(f"\n✓ Found {len(outliers)} features with suspicious ranges")

    # Now try to identify what these features represent
    print("\n" + "=" * 70)
    print("IDENTIFYING FEATURES")
    print("=" * 70)

    # Feature structure in final tensor:
    # Assume: 26 target features, then 4 nearby stations with (1 distance + 26 features + 1 soil moisture)
    # = 26 + 4 * (1 + 26 + 1) = 26 + 4*28 = 26 + 112 = 138 features

    target_features = 26  # From your output: "Feature parameters: 26"
    nearby_features_per_station = 1 + 26 + 1  # distance + features + soil
    n_nearest = 4

    print(f"\nFeature tensor structure (assuming):")
    print(f"  [0:{target_features}] = Target station features (26)")
    for i in range(n_nearest):
        start = target_features + i * nearby_features_per_station
        end = start + nearby_features_per_station
        print(f"  [{start}:{end}] = Nearby station {i+1} (distance + 26 features + soil)")

    print(f"\nMapping outlier features:")

    for outlier in outliers:
        idx = outlier['idx']

        if idx < target_features:
            # It's a target station feature
            if dense_path.exists() and idx < len(feature_params):
                param_name = feature_params[idx]
                print(f"\n  Feature {idx}: Target station - {param_name}")
                print(f"    Range: [{outlier['min']:.2f}, {outlier['max']:.2f}]")

                if timeseries_df is not None:
                    check_parameter_in_data(timeseries_df, param_name)
        else:
            # It's a nearby station feature
            offset = idx - target_features
            station_num = offset // nearby_features_per_station
            position_in_station = offset % nearby_features_per_station

            if position_in_station == 0:
                print(f"\n  Feature {idx}: Nearby station {station_num+1} - DISTANCE")
            elif position_in_station <= target_features:
                param_idx = position_in_station - 1
                if dense_path.exists() and param_idx < len(feature_params):
                    param_name = feature_params[param_idx]
                    print(f"\n  Feature {idx}: Nearby station {station_num+1} - {param_name}")
                    print(f"    Range: [{outlier['min']:.2f}, {outlier['max']:.2f}]")

                    if timeseries_df is not None:
                        check_parameter_in_data(timeseries_df, param_name)
            else:
                print(f"\n  Feature {idx}: Nearby station {station_num+1} - SOIL MOISTURE")

    # Check how -9999 is being compared
    print("\n" + "=" * 70)
    print("CHECKING INVALID MARKER DETECTION")
    print("=" * 70)

    if timeseries_df is not None:
        # Check for values close to -9999 but not exact
        almost_9999 = timeseries_df[
            (timeseries_df['value'] < -9998) &
            (timeseries_df['value'] > -10000)
        ]
        if not almost_9999.empty:
            print(f"\n⚠️  Found {len(almost_9999)} values near -9999 (but not exact):")
            print(almost_9999['value'].unique()[:20])
            print("  These might not be filtered by exact == comparison!")

def check_parameter_in_data(timeseries_df, param_name):
    """Check actual data distribution for a parameter"""
    param_data = timeseries_df[timeseries_df['parameter_code'] == param_name]['value']

    if param_data.empty:
        print(f"    ⚠️  No data found in timeseries!")
        return

    # Exclude obvious invalid markers (with tolerance)
    valid_data = param_data[
        (param_data > -9000) &  # Not -9999
        (param_data > -1500)    # Not -1000
    ]

    if len(valid_data) < len(param_data) * 0.1:
        print(f"    ⚠️  Most data is invalid markers!")
        return

    print(f"    Actual data distribution ({len(valid_data)} valid values):")
    print(f"      Min: {valid_data.min():.2f}")
    print(f"      Max: {valid_data.max():.2f}")
    print(f"      Median: {valid_data.median():.2f}")
    print(f"      95th percentile: {valid_data.quantile(0.95):.2f}")
    print(f"      99.9th percentile: {valid_data.quantile(0.999):.2f}")

    # Check if cumulative
    if param_name.startswith('PP_'):
        print(f"    ⚠️  Precipitation parameter - could be cumulative!")
        if valid_data.max() > 10000:
            print(f"       {valid_data.max():.0f}mm = {valid_data.max()/1000:.1f} meters (DEFINITELY cumulative!)")

if __name__ == "__main__":
    find_all_outliers()
