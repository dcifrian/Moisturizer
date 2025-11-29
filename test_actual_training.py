#!/usr/bin/env python3
"""
Mirror EXACTLY what happens in __main2__ with the ACTUAL precomputed dataset
This tests the splitting code path which is suspect
"""

import numpy as np
import torch
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

print("=" * 70)
print("MIRRORING ACTUAL TRAINING FLOW (__main2__)")
print("=" * 70)

# EXACTLY like loadDataset() does
collector = MeteoGaliciaCollector()

print("\n" + "=" * 70)
print("STEP 1: Loading dataset (like loadDataset())")
print("=" * 70)

# Get filtered parameters - THIS IS THE KEY!
# loadDataset() calls analyze_parameter_coverage() which uses ml_ready_dataset.csv
# But ml_ready_dataset.csv doesn't exist, so this will fail
print("\nTrying to get filtered_params from ml_ready_dataset.csv...")
try:
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
    print(f"✓ Got {len(filtered_params)} parameters from ml_ready_dataset.csv")
    print(f"Parameters: {filtered_params[:5]}...")
except Exception as e:
    print(f"✗ Failed to load from ml_ready_dataset.csv: {e}")
    print(f"\n⚠️  loadDataset() would FAIL here!")
    print(f"  But let's continue with feature_params=None (defaults to ALL)")
    filtered_params = None

# Load dataset like loadDataset() does
precomputed_path = collector.data_dir / "precomputed_sequences.npz"
norm_stats_path = collector.data_dir / "normalization_stats.npz"

print(f"\nLoading dataset...")
print(f"  Precomputed: {precomputed_path}")
print(f"  Norm stats: {norm_stats_path}")
print(f"  feature_params: {filtered_params if filtered_params else 'None (will default to ALL params!)'}")

dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=64,
    n_nearest=4,
    feature_params=filtered_params,  # This could be None!
    precomputed_path=str(precomputed_path),
    normalize=True,
    norm_stats_path=str(norm_stats_path)
)

print(f"\n✓ Dataset loaded")
print(f"  Total samples: {len(dataset)}")
print(f"  feature_params: {len(dataset.feature_params)}")
print(f"  Is pre-normalized: {dataset.is_prenormalized}")

if dataset.precomputed_data is not None:
    precomp_shape = dataset.precomputed_data['features'].shape
    print(f"  Precomputed shape: {precomp_shape}")

    # Check for mismatch
    target_feat_count = len(dataset.feature_params)
    nearby_feat_count = 1 + len(dataset.feature_params) + 1
    expected_total = target_feat_count + nearby_feat_count * 4
    actual_total = precomp_shape[2]

    print(f"\n  Feature count check:")
    print(f"    dataset.feature_params: {target_feat_count}")
    print(f"    Expected total features: {expected_total}")
    print(f"    Precomputed total features: {actual_total}")

    if actual_total != expected_total:
        print(f"\n  ❌ MISMATCH! This WILL cause problems!")
        print(f"     Precomputed was built with different feature_params!")
    else:
        print(f"  ✓ Feature counts match")

# STEP 2: Split like __main2__ does
print("\n" + "=" * 70)
print("STEP 2: Creating train/val/test splits (like __main2__)")
print("=" * 70)

train_ds, val_ds, test_ds = SoilMoistureSequenceDataset.train_val_test_split(
    dataset,
    val_stations_ratio=0.15,
    test_stations_ratio=0.0
)

print(f"\n✓ Splits created:")
print(f"  Train: {len(train_ds)} samples")
print(f"  Val: {len(val_ds) if val_ds else 0} samples")

# Check train_ds configuration
print(f"\nTrain dataset config:")
print(f"  feature_params: {len(train_ds.feature_params)}")
print(f"  Is pre-normalized: {train_ds.is_prenormalized}")

# STEP 3: Check for leakage (EXACTLY like __main2__)
print("\n" + "=" * 70)
print("STEP 3: Data leakage check (EXACTLY like __main2__)")
print("=" * 70)

# Get first sample from train_ds (after splitting!)
sample = train_ds[0]
features = sample['features']  # [64, total_features]
target = sample['target']  # [1]

print(f"\n=== Data Leakage Check ===")
print(f"Feature shape: {features.shape}")
print(f"Target value: {target.item():.4f}")

# EXACTLY like __main2__: Check if target appears in features
features_np = features.numpy()
matches = (np.abs(features_np - target.item()) < 0.001).sum()
print(f"Features matching target value: {matches}")

if matches > 0:
    print("⚠️  LEAKAGE DETECTED: Target value found in features!")
else:
    print("✓ No obvious leakage detected")

# Check last timestep (like __main2__)
last_timestep = features_np[-1, :]
last_matches = (np.abs(last_timestep - target.item()) < 0.001).sum()
print(f"Last timestep matches: {last_matches}")

# ADDITIONAL: Check only target station features (first 40 like user's code)
print(f"\n" + "=" * 70)
print(f"ADDITIONAL: Check only first 40 features (target station)")
print(f"=" * 70)

features_np_40 = features_np[:, :40]
print(f"Shape of first 40 features: {features_np_40.shape}")
matches_40 = (np.abs(features_np_40 - target.item()) < 0.001).sum()
print(f"Matches in first 40 features: {matches_40}")

if matches_40 > 0:
    print(f"❌ LEAKAGE in first 40 features!")

    # Find where
    match_mask = (np.abs(features_np_40 - target.item()) < 0.001)
    match_positions = np.where(match_mask)
    timesteps = match_positions[0]
    feat_indices = match_positions[1]

    print(f"\nLeak locations:")
    for i in range(min(5, len(timesteps))):
        t = timesteps[i]
        f = feat_indices[i]
        val = features_np_40[t, f]
        print(f"  [timestep={t}, feature={f}] = {val:.6f}")

    # Denormalize to check if real
    if train_ds.norm_stats is not None:
        print(f"\nDenormalizing first match...")
        target_min = train_ds.norm_stats['target_min']
        target_max = train_ds.norm_stats['target_max']

        t = timesteps[0]
        f = feat_indices[0]

        feat_min = train_ds.norm_stats['feature_mins'][f]
        feat_max = train_ds.norm_stats['feature_maxs'][f]

        target_raw = (target.item() + 1.0) / 2.0 * (target_max - target_min) + target_min
        feature_raw = (features_np_40[t, f] + 1.0) / 2.0 * (feat_max - feat_min) + feat_min

        print(f"  Target (raw): {target_raw:.6f}")
        print(f"  Feature {f} (raw): {feature_raw:.6f}")
        print(f"  Raw difference: {abs(target_raw - feature_raw):.6f}")

        if abs(target_raw - feature_raw) < 0.01:
            print(f"\n❌❌❌ RAW VALUES MATCH! REAL DATA LEAKAGE!")
        else:
            print(f"\n✓ Raw values differ - false positive")
else:
    print(f"✓ No leakage in first 40!")

# Check multiple samples
print(f"\n" + "=" * 70)
print(f"STEP 4: Check first 10 train samples")
print(f"=" * 70)

leak_count = 0
for idx in range(min(10, len(train_ds))):
    sample = train_ds[idx]
    features = sample['features'].numpy()
    target = sample['target'].item()

    # Check first 40 like user's code
    matches = (np.abs(features[:, :40] - target) < 0.001).sum()

    if matches > 0:
        leak_count += 1
        print(f"  Sample {idx}: {matches} matches in first 40 - LEAK!")
    else:
        print(f"  Sample {idx}: OK")

if leak_count > 0:
    print(f"\n❌ Found leakage in {leak_count}/10 samples")
else:
    print(f"\n✓ All samples clean!")
