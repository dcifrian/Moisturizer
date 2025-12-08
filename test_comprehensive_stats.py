#!/usr/bin/env python3
"""
Test script to compare old vs new normalization stats computation.
Builds a small dataset and compares stats to identify bugs.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import the main module
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset, buildDataset

def main():
    print("=" * 70)
    print("TESTING COMPREHENSIVE STATS vs OLD AUGMENTED STATS")
    print("=" * 70)

    # Build a small dataset
    print("\n1. Building small test dataset...")
    collector = MeteoGaliciaCollector()

    # Check if data exists
    if not collector.timeseries_file.exists():
        print("   No cached data found. Building small dataset (4 days, seq_length=2)...")
        # This will download data - might take a while
        buildDataset(seq_length=2, days=30, force_refresh=True)
    else:
        print("   Using cached data")

    # Load the raw data
    print("\n2. Loading raw data...")
    stations_df = pd.read_csv(collector.stations_file)
    timeseries_df = pd.read_csv(collector.timeseries_file)
    timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])
    nearest_df = pd.read_csv(collector.nearest_file)

    print(f"   Stations: {len(stations_df)}")
    print(f"   Timeseries records: {len(timeseries_df)}")

    # Analyze parameter coverage
    print("\n3. Analyzing parameter coverage...")
    _, filtered_params = collector.analyze_parameter_coverage(
        timeseries_df=timeseries_df,
        stations_df=stations_df,
        coverage_threshold=0.25
    )
    print(f"   Filtered params: {len(filtered_params)}")

    # Build dense arrays
    print("\n4. Building dense arrays...")
    dense_path = collector.data_dir / "dense_features_test.npz"
    if not dense_path.exists():
        collector.build_dense_feature_arrays(
            timeseries_df, stations_df, filtered_params,
            output_path=str(dense_path)
        )

    # Create dataset with n_nearest=5 for augmentation testing
    print("\n5. Creating dataset with n_nearest=5...")

    # First, we need nearest_df with 5 neighbors
    # Check if current nearest_df has enough neighbors
    n_cols = len([c for c in nearest_df.columns if c.startswith('nearest_') and c.endswith('_id')])
    print(f"   Current nearest_df has {n_cols} neighbors")

    if n_cols < 5:
        print("   Recalculating nearest stations with n_nearest=5...")
        nearest_df = collector.calculate_nearest_stations(stations_df, n_nearest=5)

    # Create base dataset
    dataset = SoilMoistureSequenceDataset(
        timeseries=timeseries_df,
        stations=stations_df,
        nearest=nearest_df,
        seq_length=2,
        n_nearest=5,
        feature_params=filtered_params,
        normalize=False,
        dense_array_path=str(dense_path)
    )

    print(f"   Dataset samples: {len(dataset)}")

    # Precompute and save with old stats computation
    print("\n6. Precomputing with OLD stats method (from precomputed data)...")
    precomputed_path = collector.data_dir / "precomputed_test_5nearby"
    norm_stats_path = collector.data_dir / "normalization_stats_test.npz"

    dataset.precompute_and_save(
        output_path=str(precomputed_path),
        normalize=True,
        norm_stats_path=str(norm_stats_path)
    )

    # Load the saved stats
    print("\n7. Loading saved stats...")
    saved_stats = np.load(norm_stats_path, allow_pickle=True)

    print("\n   Saved stats keys:", list(saved_stats.keys()))
    print(f"   feature_mins shape: {saved_stats['feature_mins'].shape}")
    print(f"   nearby_slot_mins shape: {saved_stats['nearby_slot_mins'].shape}")
    print(f"   n_nearby_slots: {saved_stats['n_nearby_slots']}")

    # Now test the expansion
    print("\n8. Testing expand_canonical_to_augmented_stats...")
    from create_moisture_map import expand_canonical_to_augmented_stats

    n_params = len(filtered_params)

    # Expand for 4 nearby in features, 5 available (augmented)
    expanded_aug = expand_canonical_to_augmented_stats(
        saved_stats, n_params,
        n_nearby_in_features=4,
        n_nearby_available=5,
        augmented=True
    )

    print(f"\n   Expanded feature_mins shape: {expanded_aug['feature_mins'].shape}")
    print(f"   Expanded feature_maxs shape: {expanded_aug['feature_maxs'].shape}")

    # Compare the precomputed feature_mins/maxs (which are for 5 nearby)
    # with expanded (which should match if expansion is correct)
    print("\n9. Comparing stats...")

    # The saved feature_mins/maxs are for 5 nearby (full precomputed layout)
    # We need to compare with expanded for 4 nearby

    # Actually, let's compare the per-slot stats directly
    print("\n   Per-slot nearby stats (from comprehensive):")
    nearby_slot_mins = saved_stats['nearby_slot_mins']
    nearby_slot_maxs = saved_stats['nearby_slot_maxs']

    for slot in range(min(5, nearby_slot_mins.shape[0])):
        print(f"   Slot {slot}: distance min={nearby_slot_mins[slot, 0]:.1f}, max={nearby_slot_maxs[slot, 0]:.1f}")

    # Compare with what we'd compute from the precomputed feature vector
    feature_mins = saved_stats['feature_mins']
    feature_maxs = saved_stats['feature_maxs']

    nearby_features_per_station = 1 + n_params + 1  # distance + params + soil

    print("\n   Distance stats from precomputed feature vector:")
    for slot in range(5):
        start_idx = n_params + slot * nearby_features_per_station
        dist_min = feature_mins[start_idx]
        dist_max = feature_maxs[start_idx]
        print(f"   Slot {slot}: distance min={dist_min:.1f}, max={dist_max:.1f}")

    # Check if they match
    print("\n10. Checking for mismatches...")

    mismatches = []
    for slot in range(5):
        start_idx = n_params + slot * nearby_features_per_station

        # Distance (feature 0 in nearby)
        precomputed_dist_min = feature_mins[start_idx]
        precomputed_dist_max = feature_maxs[start_idx]
        comprehensive_dist_min = nearby_slot_mins[slot, 0]
        comprehensive_dist_max = nearby_slot_maxs[slot, 0]

        if abs(precomputed_dist_min - comprehensive_dist_min) > 0.01:
            mismatches.append(f"Slot {slot} dist min: precomputed={precomputed_dist_min:.1f}, comprehensive={comprehensive_dist_min:.1f}")
        if abs(precomputed_dist_max - comprehensive_dist_max) > 0.01:
            mismatches.append(f"Slot {slot} dist max: precomputed={precomputed_dist_max:.1f}, comprehensive={comprehensive_dist_max:.1f}")

    if mismatches:
        print("\n   *** MISMATCHES FOUND ***")
        for m in mismatches:
            print(f"   {m}")
    else:
        print("\n   ✓ All stats match!")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
