#!/usr/bin/env python3
"""
Diagnose data leakage issue - check if target value appears in TARGET STATION features
(Nearby station soil moisture is expected and NOT considered leakage)
"""

import numpy as np
import torch
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

collector = MeteoGaliciaCollector()

print("=" * 70)
print("CHECKING FOR DATA LEAKAGE")
print("=" * 70)

# Check what data files exist - support both .npz and .npy directory
precomp_npz = collector.data_dir / "precomputed_sequences.npz"
precomp_npy = collector.data_dir / "precomputed_sequences"

if precomp_npy.exists() and precomp_npy.is_dir():
    precomp_path = precomp_npy
    print(f"\nUsing .npy directory: {precomp_path}")
    # Load from directory
    features = np.load(precomp_path / "features.npy", mmap_mode='r')
    targets = np.load(precomp_path / "targets.npy", mmap_mode='r')
    print(f"  Samples: {len(targets)}")
    print(f"  Features shape: {features.shape}")
    print(f"  Seq length: {features.shape[1]}")
elif precomp_npz.exists():
    precomp_path = precomp_npz
    print(f"\nUsing .npz file: {precomp_path}")
    data = np.load(precomp_path)
    print(f"  Samples: {len(data['targets'])}")
    print(f"  Features shape: {data['features'].shape}")
    print(f"  Seq length: {data['features'].shape[1]}")
else:
    print(f"\n⚠️  No precomputed sequences found!")
    exit(1)

# Load dataset
print("\nLoading dataset...")
dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=64,
    n_nearest=4,
    precomputed_path=str(precomp_path),
    normalize=True,
    norm_stats_path=str(collector.data_dir / "normalization_stats.npz")
)

print(f"\nDataset loaded: {len(dataset)} samples")
print(f"Feature params: {len(dataset.feature_params)}")
print(f"Soil moisture param: {dataset.soil_moisture_param}")
print(f"Soil in feature_params? {dataset.soil_moisture_param in dataset.feature_params}")

# Calculate feature structure
target_feat_count = len(dataset.feature_params)
nearby_feat_count = 1 + len(dataset.feature_params) + 1  # distance + weather + soil

print(f"\n" + "=" * 70)
print("FEATURE STRUCTURE")
print("=" * 70)
print(f"Target station features: indices 0-{target_feat_count-1} (count: {target_feat_count})")
print(f"  → These should NOT contain soil moisture (would be leakage!)")
print(f"Per nearby station: {nearby_feat_count} features each")
print(f"  → Distance + {len(dataset.feature_params)} weather + 1 soil moisture")
for i in range(dataset.n_nearest):
    start = target_feat_count + i * nearby_feat_count
    soil_idx = start + len(dataset.feature_params) + 1
    print(f"Nearby {i+1} soil moisture at index: {soil_idx} (expected, NOT leakage)")

print(f"\n{'='*70}")
print("CHECKING FOR LEAKAGE (target station features ONLY)")
print("=" * 70)

def check_sample_for_leakage(sample_idx, verbose=True):
    """Check if target appears in TARGET STATION features (not nearby stations)"""
    sample = dataset[sample_idx]
    features = sample['features']  # [64, total_features]
    target = sample['target'].item()
    station_id = sample['target_station_id']

    # Only check target station features (not nearby stations)
    target_features = features[:, :target_feat_count]

    # Check for matches
    matches = (torch.abs(target_features - target) < 0.0001).sum().item()

    if verbose:
        print(f"\nSample {sample_idx}:")
        print(f"  Station: {station_id}")
        print(f"  Target value: {target:.4f}")
        print(f"  Checking ONLY target station features (indices 0-{target_feat_count-1})")
        print(f"  Matches in target features: {matches}")

    if matches > 0 and verbose:
        print(f"  ⚠️  LEAKAGE DETECTED in target station features!")

        # Find positions
        match_mask = (torch.abs(target_features - target) < 0.0001)
        match_positions = torch.where(match_mask)

        print(f"  Leak positions (timestep, feature_idx):")
        for i in range(min(10, len(match_positions[0]))):
            t = match_positions[0][i].item()
            f = match_positions[1][i].item()
            val = target_features[t, f].item()

            # Get feature name
            feature_names = dataset.get_feature_names()
            feat_name = feature_names[f] if f < len(feature_names) else "unknown"

            print(f"    [{t:2d}, {f:3d}] {feat_name}: {val:.4f}")

    return matches > 0

# Check first 10 samples
print("\nChecking first 10 samples...")
leaky_samples = []
for i in range(min(10, len(dataset))):
    is_leaky = check_sample_for_leakage(i, verbose=False)
    if is_leaky:
        leaky_samples.append(i)
        print(f"  Sample {i}: LEAKAGE!")
    else:
        print(f"  Sample {i}: OK")

if leaky_samples:
    print(f"\n⚠️  {len(leaky_samples)}/10 samples have leakage in TARGET station features!")
    print(f"\nInvestigating first leaky sample in detail...")
    check_sample_for_leakage(leaky_samples[0], verbose=True)
else:
    print(f"\n✓ All checked samples are clean!")
    print(f"  (Nearby station soil moisture is expected and not counted as leakage)")

# Also check sample 3 specifically since user mentioned it
if 3 not in leaky_samples and len(dataset) > 3:
    print(f"\n{'='*70}")
    print("SPECIAL CHECK: Sample 3 (mentioned by user)")
    print("=" * 70)
    sample = dataset[3]
    features = sample['features']
    target = sample['target'].item()

    # Check ALL features (like the user's original test)
    all_matches = (torch.abs(features - target) < 0.0001).sum().item()

    # Check only target features
    target_features = features[:, :target_feat_count]
    target_matches = (torch.abs(target_features - target) < 0.0001).sum().item()

    print(f"Matches in ALL features: {all_matches}")
    print(f"Matches in target station features ONLY: {target_matches}")

    if all_matches > 0 and target_matches == 0:
        print(f"\n✓ The {all_matches} match(es) are in NEARBY station soil moisture")
        print(f"  This is EXPECTED and NOT leakage!")

        # Find which nearby stations
        nearby_features = features[:, target_feat_count:]
        nearby_matches = (torch.abs(nearby_features - target) < 0.0001)
        if nearby_matches.any():
            match_pos = torch.where(nearby_matches)
            for i in range(min(5, len(match_pos[0]))):
                t = match_pos[0][i].item()
                f = match_pos[1][i].item() + target_feat_count  # Offset to global index

                # Determine which nearby station
                nearby_idx = (f - target_feat_count) // nearby_feat_count
                pos_in_nearby = (f - target_feat_count) % nearby_feat_count

                if pos_in_nearby == len(dataset.feature_params) + 1:
                    print(f"  Match at timestep {t}, nearby station {nearby_idx+1} soil moisture (EXPECTED)")


