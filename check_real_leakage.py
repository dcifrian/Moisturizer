#!/usr/bin/env python3
"""
Properly check for data leakage by loading ACTUAL data from precomputed sequences
"""

import numpy as np
import torch
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

collector = MeteoGaliciaCollector()

print("=" * 70)
print("CHECKING ACTUAL DATA FOR LEAKAGE")
print("=" * 70)

# Load the actual dataset (the way training does it)
print("\nLoading dataset...")
dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=64,
    n_nearest=4,
    precomputed_path=str(collector.data_dir / "precomputed_sequences.npz"),
    normalize=True,
    norm_stats_path=str(collector.data_dir / "normalization_stats.npz")
)

print(f"Dataset loaded: {len(dataset)} samples")
print(f"Feature params ({len(dataset.feature_params)}): {dataset.feature_params[:5]}...")
print(f"Is pre-normalized: {dataset.is_prenormalized}")

# Calculate expected structure
target_feat_count = len(dataset.feature_params)
nearby_feat_count = 1 + len(dataset.feature_params) + 1  # distance + weather + soil
total_features = target_feat_count + nearby_feat_count * 4

print(f"\nExpected feature structure:")
print(f"  Target station features: 0-{target_feat_count-1} ({target_feat_count} features)")
print(f"  Per nearby station: {nearby_feat_count} features")
print(f"  Total: {total_features} features")

# Check first 20 samples for leakage
print(f"\n" + "=" * 70)
print(f"CHECKING SAMPLES FOR LEAKAGE")
print(f"=" * 70)

leak_count = 0
leak_details = []

for sample_idx in range(min(20, len(dataset))):
    # Get ACTUAL sample from dataset
    sample = dataset[sample_idx]

    features = sample['features']  # [seq_length, total_features]
    target = sample['target'].item()

    # Only check target station features (not nearby stations)
    target_station_features = features[:, :target_feat_count]

    # Check for matches
    matches_mask = (torch.abs(target_station_features - target) < 0.001)
    num_matches = matches_mask.sum().item()

    if num_matches > 0:
        leak_count += 1

        # Find where the matches are
        match_positions = torch.where(matches_mask)
        timesteps = match_positions[0].numpy()
        feat_indices = match_positions[1].numpy()

        print(f"\nSample {sample_idx}: {num_matches} match(es) - LEAK!")
        print(f"  Target value: {target:.6f}")
        print(f"  Station ID: {sample['target_station_id']}")

        # Show first few matches
        for i in range(min(5, num_matches)):
            t = timesteps[i]
            f = feat_indices[i]
            val = target_station_features[t, f].item()

            # Get feature name
            if f < len(dataset.feature_params):
                feat_name = dataset.feature_params[f]
            else:
                feat_name = f"unknown_{f}"

            diff = abs(val - target)
            print(f"    [timestep={t}, feature={f}] {feat_name} = {val:.6f} (diff: {diff:.6f})")

        leak_details.append({
            'sample_idx': sample_idx,
            'station_id': sample['target_station_id'],
            'target': target,
            'num_matches': num_matches,
            'first_match_timestep': timesteps[0],
            'first_match_feature': feat_indices[0]
        })
    else:
        print(f"Sample {sample_idx}: OK")

print(f"\n" + "=" * 70)
print(f"RESULTS")
print(f"=" * 70)

if leak_count > 0:
    print(f"\n❌ LEAKAGE DETECTED in {leak_count} samples!")
    print(f"\nLeaky samples summary:")
    for detail in leak_details:
        print(f"  Sample {detail['sample_idx']}: station {detail['station_id']}, "
              f"{detail['num_matches']} matches at feature {detail['first_match_feature']}")

    # Investigate the first leak in detail
    if leak_details:
        print(f"\n" + "=" * 70)
        print(f"DETAILED INVESTIGATION OF FIRST LEAK")
        print(f"=" * 70)

        first_leak = leak_details[0]
        sample_idx = first_leak['sample_idx']
        sample = dataset[sample_idx]

        features = sample['features']
        target = sample['target'].item()

        leak_timestep = first_leak['first_match_timestep']
        leak_feature = first_leak['first_match_feature']
        leak_value = features[leak_timestep, leak_feature].item()

        print(f"\nSample {sample_idx}:")
        print(f"  Target (normalized): {target:.6f}")
        print(f"  Leaked feature index: {leak_feature}")
        print(f"  Leaked feature name: {dataset.feature_params[leak_feature]}")
        print(f"  Leaked value (normalized): {leak_value:.6f}")
        print(f"  Difference: {abs(leak_value - target):.6f}")

        # Try to denormalize if we have stats
        if dataset.norm_stats is not None:
            target_min = dataset.norm_stats['target_min']
            target_max = dataset.norm_stats['target_max']
            feat_min = dataset.norm_stats['feature_mins'][leak_feature]
            feat_max = dataset.norm_stats['feature_maxs'][leak_feature]

            # Denormalize: norm = 2.0 * (raw - min) / (max - min) - 1.0
            # So: raw = (norm + 1.0) / 2.0 * (max - min) + min
            target_raw = (target + 1.0) / 2.0 * (target_max - target_min) + target_min
            feature_raw = (leak_value + 1.0) / 2.0 * (feat_max - feat_min) + feat_min

            print(f"\nDenormalized values:")
            print(f"  Target (soil moisture): {target_raw:.6f} (range: [{target_min:.2f}, {target_max:.2f}])")
            print(f"  Feature ({dataset.feature_params[leak_feature]}): {feature_raw:.6f} (range: [{feat_min:.2f}, {feat_max:.2f}])")
            print(f"  Raw difference: {abs(target_raw - feature_raw):.6f}")

            if abs(target_raw - feature_raw) < 0.01:
                print(f"\n❌❌❌ RAW VALUES ARE NEARLY IDENTICAL!")
                print(f"  This is REAL DATA LEAKAGE!")
                print(f"  The feature contains actual soil moisture instead of {dataset.feature_params[leak_feature]}!")
            else:
                print(f"\n✓ Raw values are different")
                print(f"  This is a coincidence - normalized values happen to match")
                print(f"  Feature is correctly {dataset.feature_params[leak_feature]}, not soil moisture")

else:
    print(f"\n✓ No leakage detected in checked samples!")
    print(f"  All target station features are clean.")
