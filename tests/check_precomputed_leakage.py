#!/usr/bin/env python3
"""
Check if precomputed sequences have data leakage
"""

import numpy as np
import torch
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector

collector = MeteoGaliciaCollector()

print("=" * 70)
print("CHECKING PRECOMPUTED SEQUENCES FOR LEAKAGE")
print("=" * 70)

# Load precomputed sequences (.npy directory)
precomp_dir = collector.data_dir / "precomputed_sequences"

if not precomp_dir.exists() or not precomp_dir.is_dir():
    print(f"\n✗ Precomputed .npy directory not found: {precomp_dir}")
    print("  Checking for .npz file...")
    precomp_npz = collector.data_dir / "precomputed_sequences.npz"
    if precomp_npz.exists():
        print(f"  Loading from .npz: {precomp_npz}")
        data = np.load(precomp_npz)
        features = data['features']
        targets = data['targets']
    else:
        print(f"  ✗ No precomputed data found")
        exit(1)
else:
    print(f"\nLoading from .npy directory: {precomp_dir}")
    features = np.load(precomp_dir / "features.npy", mmap_mode='r')
    targets = np.load(precomp_dir / "targets.npy", mmap_mode='r')

print(f"  Features shape: {features.shape}")
print(f"  Targets shape: {targets.shape}")

# Features shape: [num_samples, seq_length, total_features]
# total_features = target_features + (nearby_features * 4)
# If we have 23 feature_params (no soil moisture):
#   target_features = 23
#   nearby_features = 1 (distance) + 23 (weather) + 1 (soil) = 25
#   total = 23 + 25*4 = 23 + 100 = 123

num_samples, seq_length, total_features = features.shape
print(f"\nFeature analysis:")
print(f"  Total features per timestep: {total_features}")

# Calculate expected structure
# Assuming 23 filtered params (from diagnostic), 4 nearby stations
expected_target_feat = 23
expected_nearby_feat = 1 + 23 + 1  # distance + weather + soil
expected_total = expected_target_feat + (expected_nearby_feat * 4)

print(f"  Expected structure (23 params, 4 nearby):")
print(f"    Target station: 0-{expected_target_feat-1} ({expected_target_feat} features)")
print(f"    Per nearby: {expected_nearby_feat} features")
print(f"    Total expected: {expected_total}")

if total_features == expected_total:
    print(f"  ✓ Feature count matches expected!")
    target_feat_count = expected_target_feat
elif total_features == 138:
    # User mentioned 138 features from map creation
    # 138 = target + 4*nearby
    # If nearby = 1 + params + 1, then nearby = (138 - target) / 4
    # Try different target counts
    print(f"  ⚠️  Feature count is 138 (from map creation error)")
    print(f"  Trying to infer structure...")

    # Try 26 params (from user's ml_ready_dataset coverage output)
    test_target = 26
    test_nearby = 1 + 26 + 1
    test_total = test_target + test_nearby * 4
    if test_total == 138:
        print(f"  → Likely built with 26 params: target={test_target}, total={test_total}")
        target_feat_count = test_target
    else:
        print(f"  → Can't determine structure (test: {test_total} != 138)")
        target_feat_count = 40  # User said they check first 40
else:
    print(f"  ⚠️  Unexpected feature count!")
    # User said they check first 40 for target station
    target_feat_count = 40

print(f"\nChecking for leakage (target station features only: 0-{target_feat_count-1})...")

# Check first 10 samples
leak_count = 0
leak_samples = []

for sample_idx in range(min(10, num_samples)):
    target = targets[sample_idx].item() if targets.ndim > 1 else targets[sample_idx]

    # Only check target station features
    target_features = features[sample_idx, :, :target_feat_count]

    # Check for matches (same tolerance as user's code)
    matches = (np.abs(target_features - target) < 0.001).sum()

    if matches > 0:
        leak_count += 1
        leak_samples.append(sample_idx)
        print(f"  Sample {sample_idx}: {matches} match(es) - LEAK!")

        # Find where
        match_mask = (np.abs(target_features - target) < 0.001)
        timesteps, feat_indices = np.where(match_mask)
        print(f"    Target value: {target:.4f}")
        print(f"    Matches at:")
        for i in range(min(5, len(timesteps))):
            t = timesteps[i]
            f = feat_indices[i]
            val = target_features[t, f]
            print(f"      [timestep={t}, feature={f}] = {val:.4f}")
    else:
        print(f"  Sample {sample_idx}: OK")

if leak_count > 0:
    print(f"\n❌ LEAKAGE DETECTED in {leak_count}/10 samples!")
    print(f"  Leaky samples: {leak_samples}")
    print(f"\nThe precomputed sequences contain data leakage.")
    print(f"This means the leak happened during precomputation, NOT during training.")
    print(f"\nPossible causes:")
    print(f"  1. Precomputed with wrong feature_params (included soil moisture)")
    print(f"  2. Bug in sequence building during precomputation")
    print(f"  3. Precomputed data is stale (built with old buggy code)")
else:
    print(f"\n✓ No leakage detected in checked samples!")
    print(f"  All target station features are clean.")

# Also check sample 3 specifically (user mentioned it)
if num_samples > 3:
    print(f"\n" + "=" * 70)
    print(f"SPECIAL CHECK: Sample 3")
    print(f"=" * 70)

    target = targets[3].item() if targets.ndim > 1 else targets[3]
    sample_features = features[3]  # [seq_length, total_features]

    # Check ALL features
    all_matches = (np.abs(sample_features - target) < 0.001).sum()

    # Check only target station features
    target_station_features = sample_features[:, :target_feat_count]
    target_matches = (np.abs(target_station_features - target) < 0.001).sum()

    print(f"Target value: {target:.4f}")
    print(f"Matches in ALL features: {all_matches}")
    print(f"Matches in target station features ONLY: {target_matches}")

    if all_matches > 0 and target_matches == 0:
        print(f"\n✓ The {all_matches} match(es) are in NEARBY station soil moisture")
        print(f"  This is EXPECTED and NOT leakage!")
    elif target_matches > 0:
        print(f"\n❌ LEAKAGE: {target_matches} match(es) in target station features!")
