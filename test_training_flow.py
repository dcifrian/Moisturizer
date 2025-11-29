#!/usr/bin/env python3
"""
Mirror the EXACT training flow from __main2__ with 16 days to test for leakage
This tests what actually happens during training, including the split
"""

import numpy as np
import torch
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset, buildDataset

print("=" * 70)
print("TESTING ACTUAL TRAINING FLOW (16 days)")
print("=" * 70)

# Step 1: Build small dataset (16 days instead of 3705)
print("\n" + "=" * 70)
print("STEP 1: Building dataset with 16 days")
print("=" * 70)

train_ds, val_ds, test_ds = buildDataset(seq_length=64, days=16)

if len(train_ds) == 0:
    print("\n✗ No training samples! Dataset too small.")
    print("  This is expected with only 16 days (need 64 days history + data)")
    exit(0)

print(f"\n✓ Dataset built successfully")
print(f"  Train: {len(train_ds)} samples")
print(f"  Val: {len(val_ds)} samples" if val_ds else "  Val: None")

# Step 2: Check the dataset configuration
print("\n" + "=" * 70)
print("STEP 2: Dataset configuration check")
print("=" * 70)

print(f"\nTrain dataset:")
print(f"  feature_params: {len(train_ds.feature_params)}")
print(f"  Parameters: {train_ds.feature_params}")
print(f"  Using precomputed: {train_ds.precomputed_data is not None}")
print(f"  Pre-normalized: {train_ds.is_prenormalized}")

if train_ds.precomputed_data is not None:
    precomp_shape = train_ds.precomputed_data['features'].shape
    print(f"  Precomputed features shape: {precomp_shape}")
    print(f"    Expected: [num_samples, seq_length, total_features]")

    # Calculate expected feature count from dataset config
    target_feat_count = len(train_ds.feature_params)
    nearby_feat_count = 1 + len(train_ds.feature_params) + 1
    expected_total = target_feat_count + nearby_feat_count * 4

    print(f"  Expected total features: {expected_total}")
    print(f"    Target: {target_feat_count}")
    print(f"    Per nearby: {nearby_feat_count}")

    actual_total = precomp_shape[2]
    if actual_total != expected_total:
        print(f"\n  ⚠️  MISMATCH! Precomputed has {actual_total} but dataset expects {expected_total}")
        print(f"     This will cause indexing issues!")
    else:
        print(f"  ✓ Feature count matches")

# Step 3: Check for leakage (EXACTLY like __main2__)
print("\n" + "=" * 70)
print("STEP 3: Data leakage check (mirroring __main2__)")
print("=" * 70)

# Get first sample from TRAIN split (after splitting!)
sample = train_ds[0]
features = sample['features']  # [64, total_features]
target = sample['target']  # [1]

print(f"\n=== Data Leakage Check ===")
print(f"Feature shape: {features.shape}")
print(f"Target value: {target.item():.4f}")
print(f"Station ID: {sample['target_station_id']}")

# Check if target value appears anywhere in features (like __main2__ does)
features_np = features.numpy()
matches = (np.abs(features_np - target.item()) < 0.001).sum()
print(f"Features matching target value (ALL features): {matches}")

if matches > 0:
    print("⚠️  LEAKAGE DETECTED: Target value found in features!")

    # Find where
    match_mask = (np.abs(features_np - target.item()) < 0.001)
    match_positions = np.where(match_mask)
    timesteps = match_positions[0]
    feat_indices = match_positions[1]

    print(f"\nMatch locations:")
    for i in range(min(10, len(timesteps))):
        t = timesteps[i]
        f = feat_indices[i]
        val = features_np[t, f]
        print(f"  [timestep={t}, feature={f}] = {val:.6f}")
else:
    print("✓ No obvious leakage detected")

# Check last timestep specifically (like __main2__ does)
last_timestep = features_np[-1, :]
last_matches = (np.abs(last_timestep - target.item()) < 0.001).sum()
print(f"Last timestep matches: {last_matches}")

# Additional check: Only target station features (first 40 as user mentioned)
print(f"\n" + "=" * 70)
print(f"ADDITIONAL: Check only TARGET station features")
print(f"=" * 70)

target_feat_count = len(train_ds.feature_params)
print(f"Target station features: indices 0-{target_feat_count-1}")

target_station_features = features_np[:, :target_feat_count]
target_matches = (np.abs(target_station_features - target.item()) < 0.001).sum()

print(f"Matches in target station features ONLY: {target_matches}")

if target_matches > 0:
    print(f"❌ REAL LEAKAGE in target station features!")

    # Find where
    match_mask = (np.abs(target_station_features - target.item()) < 0.001)
    match_positions = np.where(match_mask)
    timesteps = match_positions[0]
    feat_indices = match_positions[1]

    print(f"\nTarget station leak locations:")
    for i in range(min(5, len(timesteps))):
        t = timesteps[i]
        f = feat_indices[i]
        val = target_station_features[t, f]

        # Get feature name
        feat_name = train_ds.feature_params[f] if f < len(train_ds.feature_params) else "unknown"

        print(f"  [timestep={t}, feature={f}] {feat_name} = {val:.6f}")

    # Denormalize to check if it's real or coincidence
    if train_ds.norm_stats is not None:
        print(f"\nDenormalizing to check if leak is real...")
        target_min = train_ds.norm_stats['target_min']
        target_max = train_ds.norm_stats['target_max']

        t = timesteps[0]
        f = feat_indices[0]

        feat_min = train_ds.norm_stats['feature_mins'][f]
        feat_max = train_ds.norm_stats['feature_maxs'][f]

        target_raw = (target.item() + 1.0) / 2.0 * (target_max - target_min) + target_min
        feature_raw = (target_station_features[t, f] + 1.0) / 2.0 * (feat_max - feat_min) + feat_min

        print(f"  Target (raw): {target_raw:.6f}")
        print(f"  Feature {f} (raw): {feature_raw:.6f}")
        print(f"  Raw difference: {abs(target_raw - feature_raw):.6f}")

        if abs(target_raw - feature_raw) < 0.01:
            print(f"\n❌❌❌ RAW VALUES MATCH! This is REAL DATA LEAKAGE!")
        else:
            print(f"\n✓ Raw values differ - false positive from normalization")

elif matches > 0:
    print(f"✓ Matches are in NEARBY station features (expected)")
else:
    print(f"✓ No leakage!")

# Step 4: Check multiple samples
print(f"\n" + "=" * 70)
print(f"STEP 4: Check first 10 train samples")
print(f"=" * 70)

leak_count = 0
for idx in range(min(10, len(train_ds))):
    sample = train_ds[idx]
    features = sample['features'].numpy()
    target = sample['target'].item()

    # Only check target station features
    target_station_features = features[:, :target_feat_count]
    matches = (np.abs(target_station_features - target) < 0.001).sum()

    if matches > 0:
        leak_count += 1
        print(f"  Sample {idx}: {matches} matches - LEAK!")
    else:
        print(f"  Sample {idx}: OK")

if leak_count > 0:
    print(f"\n❌ Found leakage in {leak_count}/10 samples")
else:
    print(f"\n✓ All samples clean!")
