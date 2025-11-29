#!/usr/bin/env python3
"""
Compare filtered parameters from BOTH methods and test the mismatch effect
"""

import pandas as pd
import numpy as np
import torch
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

collector = MeteoGaliciaCollector()

print("=" * 70)
print("COMPARING PARAMETER FILTERING METHODS")
print("=" * 70)

# Method 1: Direct timeseries analysis (used by buildDataset)
print("\nMethod 1: Direct timeseries analysis")
print("-" * 70)

timeseries_df = pd.read_csv(collector.timeseries_file)
stations_df = pd.read_csv(collector.stations_file)

soil_moisture_stations = stations_df[stations_df['has_soil_moisture']]['station_id'].tolist()
all_params = timeseries_df['parameter_code'].unique()
soil_moisture_param = "HS_CV_AVG_-0.2m"
coverage_threshold = 0.25
filtered_params_method1 = []

for param in sorted(all_params):
    if param == soil_moisture_param:
        continue
    param_data = timeseries_df[
        (timeseries_df['parameter_code'] == param) &
        (timeseries_df['station_id'].isin(soil_moisture_stations))
    ]
    stations_with_param = param_data['station_id'].nunique()
    coverage = stations_with_param / len(soil_moisture_stations) if soil_moisture_stations else 0

    if coverage >= coverage_threshold:
        filtered_params_method1.append(param)

# Add coordinate features - all stations have these (matching buildDataset)
coordinate_features = ['altitude', 'utmx', 'utmy']
filtered_params_method1.extend(coordinate_features)

print(f"Found {len(filtered_params_method1)} parameters")
print(f"Parameters: {sorted(filtered_params_method1)}")

# Method 2: analyze_parameter_coverage (uses ml_ready_dataset.csv)
print("\n\nMethod 2: analyze_parameter_coverage (ml_ready_dataset.csv)")
print("-" * 70)

coverage_dict, filtered_params_method2 = collector.analyze_parameter_coverage(coverage_threshold=0.25)
print(f"Found {len(filtered_params_method2)} parameters")
print(f"Parameters: {sorted(filtered_params_method2)}")

# Compare
print("\n\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

set1 = set(filtered_params_method1)
set2 = set(filtered_params_method2)

if set1 == set2:
    print("\n✓ Both methods produce IDENTICAL parameter lists")
else:
    print("\n❌ Methods produce DIFFERENT parameter lists!")

    only_method1 = sorted(set1 - set2)
    only_method2 = sorted(set2 - set1)
    common = sorted(set1 & set2)

    print(f"\nCommon: {len(common)} parameters")

    if only_method1:
        print(f"\nOnly in Method 1 (timeseries): {len(only_method1)}")
        for p in only_method1:
            print(f"  - {p}")

    if only_method2:
        print(f"\nOnly in Method 2 (ml_ready_dataset.csv): {len(only_method2)}")
        for p in only_method2:
            print(f"  - {p}")

    print(f"\nThis explains the feature count mismatch:")
    print(f"  Method 1: {len(filtered_params_method1)} params → {len(filtered_params_method1) + 4*(1 + len(filtered_params_method1) + 1)} total features")
    print(f"  Method 2: {len(filtered_params_method2)} params → {len(filtered_params_method2) + 4*(1 + len(filtered_params_method2) + 1)} total features")

# Now test what happens with the mismatch
print("\n\n" + "=" * 70)
print("TEST 1: Load with CORRECT params (method 1 - matches precomputed)")
print("=" * 70)

dataset_correct = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=64,
    n_nearest=4,
    feature_params=filtered_params_method1,  # CORRECT - matches precomputed
    precomputed_path=str(collector.data_dir / "precomputed_sequences.npz"),
    normalize=True,
    norm_stats_path=str(collector.data_dir / "normalization_stats.npz")
)

print(f"Dataset with CORRECT params:")
print(f"  feature_params: {len(dataset_correct.feature_params)}")
print(f"  Precomputed shape: {dataset_correct.precomputed_data['features'].shape if dataset_correct.precomputed_data else 'None'}")

# Split and check for leaks
train_correct, val_correct, _ = SoilMoistureSequenceDataset.train_val_test_split(
    dataset_correct, val_stations_ratio=0.15, test_stations_ratio=0.0
)

print(f"\nChecking for leaks with CORRECT params (first 10 train samples):")
leak_count_correct = 0
for idx in range(min(10, len(train_correct))):
    sample = train_correct[idx]
    features = sample['features'].numpy()
    target = sample['target'].item()

    # Check first target_feat_count features (target station)
    target_feat_count = len(dataset_correct.feature_params)
    matches = (np.abs(features[:, :target_feat_count] - target) < 0.001).sum()

    if matches > 0:
        leak_count_correct += 1
        print(f"  Sample {idx}: {matches} matches - LEAK!")

if leak_count_correct == 0:
    print(f"  ✓ All 10 samples clean!")
else:
    print(f"  ❌ Found leaks in {leak_count_correct}/10 samples")

# Test with WRONG params (method 2 or None)
print("\n\n" + "=" * 70)
print("TEST 2: Load with WRONG params (method 2 - MISMATCH)")
print("=" * 70)

dataset_wrong = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=64,
    n_nearest=4,
    feature_params=filtered_params_method2 if len(filtered_params_method2) != len(filtered_params_method1) else None,  # WRONG
    precomputed_path=str(collector.data_dir / "precomputed_sequences.npz"),
    normalize=True,
    norm_stats_path=str(collector.data_dir / "normalization_stats.npz")
)

print(f"Dataset with WRONG params:")
print(f"  feature_params: {len(dataset_wrong.feature_params)}")
print(f"  Expected total features: {len(dataset_wrong.feature_params) + 4*(1 + len(dataset_wrong.feature_params) + 1)}")
print(f"  Precomputed total features: {dataset_wrong.precomputed_data['features'].shape[2] if dataset_wrong.precomputed_data else 'None'}")

if dataset_wrong.precomputed_data:
    expected = len(dataset_wrong.feature_params) + 4*(1 + len(dataset_wrong.feature_params) + 1)
    actual = dataset_wrong.precomputed_data['features'].shape[2]
    if expected != actual:
        print(f"  ❌ MISMATCH! Expected {expected} but precomputed has {actual}")

# Split and check for leaks
train_wrong, val_wrong, _ = SoilMoistureSequenceDataset.train_val_test_split(
    dataset_wrong, val_stations_ratio=0.15, test_stations_ratio=0.0
)

print(f"\nChecking for leaks with WRONG params (first 10 train samples):")
leak_count_wrong = 0
for idx in range(min(10, len(train_wrong))):
    sample = train_wrong[idx]
    features = sample['features'].numpy()
    target = sample['target'].item()

    # Check first target_feat_count features (target station)
    target_feat_count = len(dataset_wrong.feature_params)
    # But this might be out of bounds!
    if target_feat_count <= features.shape[1]:
        matches = (np.abs(features[:, :target_feat_count] - target) < 0.001).sum()

        if matches > 0:
            leak_count_wrong += 1
            print(f"  Sample {idx}: {matches} matches - LEAK!")
    else:
        print(f"  Sample {idx}: Can't check - target_feat_count ({target_feat_count}) > features ({features.shape[1]})")

if leak_count_wrong == 0:
    print(f"  ✓ All samples clean (or couldn't check)")
else:
    print(f"  ❌ Found leaks in {leak_count_wrong}/10 samples")

# Summary
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nWith CORRECT params: {leak_count_correct}/10 samples had leaks")
print(f"With WRONG params: {leak_count_wrong}/10 samples had leaks")

if leak_count_correct > 0 and leak_count_wrong > 0:
    print(f"\n❌ Leaks occur with BOTH param sets - this is a data issue, not mismatch!")
elif leak_count_correct == 0 and leak_count_wrong > 0:
    print(f"\n❌ Leaks ONLY occur with wrong params - mismatch IS the cause!")
elif leak_count_correct > 0 and leak_count_wrong == 0:
    print(f"\n❌ Leaks ONLY occur with correct params - data issue in precomputed!")
else:
    print(f"\n✓ No leaks with either param set")
