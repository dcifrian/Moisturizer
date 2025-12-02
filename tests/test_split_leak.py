#!/usr/bin/env python3
"""
Test if the leak exists BEFORE splitting or if splitting introduces it
"""

import numpy as np
import torch
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

collector = MeteoGaliciaCollector()

# Method 1 params (correct)
import pandas as pd
timeseries_df = pd.read_csv(collector.timeseries_file)
stations_df = pd.read_csv(collector.stations_file)
soil_moisture_stations = stations_df[stations_df['has_soil_moisture']]['station_id'].tolist()
all_params = timeseries_df['parameter_code'].unique()
soil_moisture_param = "HS_CV_AVG_-0.2m"
filtered_params = []
for param in sorted(all_params):
    if param == soil_moisture_param:
        continue
    param_data = timeseries_df[
        (timeseries_df['parameter_code'] == param) &
        (timeseries_df['station_id'].isin(soil_moisture_stations))
    ]
    coverage = param_data['station_id'].nunique() / len(soil_moisture_stations)
    if coverage >= 0.25:
        filtered_params.append(param)

print("=" * 70)
print("TEST: Does leak exist BEFORE or AFTER splitting?")
print("=" * 70)

# Load base dataset
dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=64,
    n_nearest=4,
    feature_params=filtered_params,
    precomputed_path=str(collector.data_dir / "precomputed_sequences.npz"),
    normalize=True,
    norm_stats_path=str(collector.data_dir / "normalization_stats.npz")
)

print(f"\nBase dataset: {len(dataset)} samples")
target_feat_count = len(dataset.feature_params)

# Check base dataset (BEFORE splitting)
print("\n" + "-" * 70)
print("CHECKING BASE DATASET (before split)")
print("-" * 70)

leak_samples_base = []
for idx in range(min(20, len(dataset))):
    sample = dataset[idx]
    features = sample['features'].numpy()
    target = sample['target'].item()

    matches = (np.abs(features[:, :target_feat_count] - target) < 0.001).sum()

    if matches > 0:
        leak_samples_base.append(idx)
        print(f"Sample {idx}: {matches} matches - station {sample['target_station_id']}")

if leak_samples_base:
    print(f"\n❌ Found {len(leak_samples_base)} leaky samples in BASE dataset")
    print(f"Leaky indices: {leak_samples_base}")
else:
    print(f"\n✓ No leaks in base dataset")

# Now split
print("\n" + "-" * 70)
print("SPLITTING DATASET")
print("-" * 70)

train_ds, val_ds, _ = SoilMoistureSequenceDataset.train_val_test_split(
    dataset, val_stations_ratio=0.15, test_stations_ratio=0.0, random_seed=42
)

print(f"Train: {len(train_ds)} samples")
print(f"Val: {len(val_ds)} samples")

# Check train dataset (AFTER splitting)
print("\n" + "-" * 70)
print("CHECKING TRAIN DATASET (after split)")
print("-" * 70)

leak_samples_train = []
train_to_base_mapping = {}  # Map train index to base index

for train_idx in range(min(20, len(train_ds))):
    sample = train_ds[train_idx]
    features = sample['features'].numpy()
    target = sample['target'].item()
    station_id = sample['target_station_id']

    matches = (np.abs(features[:, :target_feat_count] - target) < 0.001).sum()

    if matches > 0:
        leak_samples_train.append(train_idx)

        # Find corresponding base index
        # The sample_index tells us which sample from base dataset this is
        if train_ds._indices:
            base_idx = train_ds._indices[train_idx]
            train_to_base_mapping[train_idx] = base_idx
            print(f"Train sample {train_idx} (base {base_idx}): {matches} matches - station {station_id}")
        else:
            print(f"Train sample {train_idx}: {matches} matches - station {station_id}")

if leak_samples_train:
    print(f"\n❌ Found {len(leak_samples_train)} leaky samples in TRAIN dataset")
    print(f"Leaky train indices: {leak_samples_train}")
else:
    print(f"\n✓ No leaks in train dataset")

# Analysis
print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

if leak_samples_base and leak_samples_train:
    print("\n❌ Leaks exist in BOTH base and train datasets")
    print("   The leak is in the precomputed data itself, NOT introduced by splitting")

    # Check if same samples
    if train_to_base_mapping:
        print(f"\nMapping train leaks to base indices:")
        for train_idx, base_idx in train_to_base_mapping.items():
            in_base = "YES" if base_idx in leak_samples_base else "NO"
            print(f"  Train {train_idx} → Base {base_idx}: In base leaks? {in_base}")

elif not leak_samples_base and leak_samples_train:
    print("\n❌ Leaks ONLY in train dataset!")
    print("   Splitting IS introducing the leak!")

elif leak_samples_base and not leak_samples_train:
    print("\n⚠️  Leaks in base but NOT in train")
    print("   The leaky samples weren't selected for train split")

else:
    print("\n✓ No leaks found in either dataset")

# Check a specific leaky sample in detail
if leak_samples_base:
    print("\n" + "=" * 70)
    print(f"DETAILED CHECK: Base sample {leak_samples_base[0]}")
    print("=" * 70)

    sample = dataset[leak_samples_base[0]]
    features = sample['features'].numpy()
    target = sample['target'].item()

    print(f"Target: {target:.6f}")
    print(f"Station: {sample['target_station_id']}")
    print(f"Features shape: {features.shape}")

    # Find the leak
    match_mask = (np.abs(features[:, :target_feat_count] - target) < 0.001)
    match_positions = np.where(match_mask)

    for i in range(min(3, len(match_positions[0]))):
        t = match_positions[0][i]
        f = match_positions[1][i]
        val = features[t, f]
        param_name = filtered_params[f] if f < len(filtered_params) else "unknown"
        print(f"  Match at [t={t}, f={f}] {param_name}: {val:.6f}")
