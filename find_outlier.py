#!/usr/bin/env python3
"""
Find the feature with the 71639 outlier in normalization stats
"""

import numpy as np
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector
import pandas as pd

def find_outlier_feature():
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
    print("FINDING 71639 OUTLIER")
    print("=" * 70)

    # Find the feature with max ~71639
    target_value = 71639.453

    for feat_idx in range(len(feature_maxs)):
        if abs(feature_maxs[feat_idx] - target_value) < 100:  # Within 100 of target
            print(f"\n✓ FOUND IT!")
            print(f"  Feature index: {feat_idx}")
            print(f"  Min: {feature_mins[feat_idx]:.2f}")
            print(f"  Max: {feature_maxs[feat_idx]:.2f}")
            print(f"  Range: {feature_maxs[feat_idx] - feature_mins[feat_idx]:.2f}")

            # Load dense arrays to get feature name
            dense_path = collector.data_dir / "dense_features.npz"
            if dense_path.exists():
                dense_data = np.load(dense_path)
                feature_params = dense_data['feature_params'].tolist()
                if feat_idx < len(feature_params):
                    param_name = feature_params[feat_idx]
                    print(f"  Parameter name: {param_name}")

                    # Search raw timeseries for this outlier
                    print(f"\n  Searching raw timeseries for outlier values...")
                    timeseries_df = pd.read_csv(collector.timeseries_file)

                    outlier_data = timeseries_df[
                        (timeseries_df['parameter_code'] == param_name) &
                        (timeseries_df['value'] > 10000)  # Suspiciously high
                    ]

                    if not outlier_data.empty:
                        print(f"\n  Found {len(outlier_data)} outlier values for {param_name}:")
                        print(outlier_data[['station_id', 'date', 'value']].head(10))

                        print(f"\n  Value distribution for {param_name}:")
                        param_data = timeseries_df[timeseries_df['parameter_code'] == param_name]['value']
                        print(f"    Min: {param_data.min():.2f}")
                        print(f"    Max: {param_data.max():.2f}")
                        print(f"    Median: {param_data.median():.2f}")
                        print(f"    Mean: {param_data.mean():.2f}")
                        print(f"    95th percentile: {param_data.quantile(0.95):.2f}")
                        print(f"    99th percentile: {param_data.quantile(0.99):.2f}")
                        print(f"    99.9th percentile: {param_data.quantile(0.999):.2f}")

                        # Check if this could be cumulative
                        if param_name.startswith('PP_'):
                            print(f"\n  ⚠️  This is a precipitation parameter (PP_*)!")
                            print(f"     Could be cumulative precipitation (mm over long period)")
                            print(f"     71639mm = 71.6 meters of rain (clearly cumulative!)")

            return feat_idx

    print("\n✗ Feature not found")
    return None

if __name__ == "__main__":
    find_outlier_feature()
