#!/usr/bin/env python3
"""
Diagnose data leakage issue - check if target value appears in features
"""

import numpy as np
import torch
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

collector = MeteoGaliciaCollector()

print("=" * 70)
print("CHECKING FOR DATA LEAKAGE")
print("=" * 70)

# Check what data files exist
import os
precomp_path = collector.data_dir / "precomputed_sequences.npz"
if precomp_path.exists():
    data = np.load(precomp_path)
    print(f"\nPrecomputed sequences file: {precomp_path}")
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

# Get first sample
sample = dataset[0]
features = sample['features']
target = sample['target'].item()
station_id = sample['target_station_id']

print(f"\n" + "=" * 70)
print("SAMPLE 0 ANALYSIS")
print("=" * 70)
print(f"Target station: {station_id}")
print(f"Features shape: {features.shape}")
print(f"Target value: {target:.4f}")

# Check for matches
matches = (torch.abs(features - target) < 0.0001).sum().item()
print(f"Features matching target: {matches}")

# Check last timestep
last_timestep = features[-1, :]
last_matches = (torch.abs(last_timestep - target) < 0.0001).sum().item()
print(f"Last timestep matches: {last_matches}")

if matches > 0:
    print(f"\n⚠️  LEAKAGE DETECTED!")

    # Find match positions
    match_mask = (torch.abs(features - target) < 0.0001)
    match_positions = torch.where(match_mask)

    print(f"\nMatch positions (timestep, feature_idx, value):")
    for i in range(min(20, len(match_positions[0]))):
        t = match_positions[0][i].item()
        f = match_positions[1][i].item()
        val = features[t, f].item()
        print(f"  [{t:2d}, {f:3d}] = {val:.4f}")

    # Group by timestep
    timesteps = match_positions[0].unique()
    print(f"\nMatches by timestep:")
    for t in timesteps[:10]:
        count = (match_positions[0] == t).sum().item()
        print(f"  Timestep {t.item():2d}: {count} matches")

    # Group by feature index
    feature_indices = match_positions[1].unique()
    print(f"\nLeaking feature indices: {feature_indices.tolist()}")

    # Map to feature names
    feature_names = dataset.get_feature_names()
    print(f"\nLeaking features:")
    for idx in feature_indices:
        idx_val = idx.item()
        if idx_val < len(feature_names):
            print(f"  [{idx_val:3d}] {feature_names[idx_val]}")

    # Analyze structure
    print(f"\n" + "=" * 70)
    print("FEATURE STRUCTURE ANALYSIS")
    print("=" * 70)

    target_feat_count = len(dataset.feature_params)
    nearby_feat_count = 1 + len(dataset.feature_params) + 1  # distance + weather + soil

    print(f"Target station features: 0-{target_feat_count-1} (count: {target_feat_count})")
    for i in range(dataset.n_nearest):
        start = target_feat_count + i * nearby_feat_count
        end = start + nearby_feat_count - 1
        print(f"Nearby {i+1} features: {start}-{end}")
        print(f"  Distance: {start}")
        print(f"  Weather: {start+1}-{start+len(dataset.feature_params)}")
        print(f"  Soil moisture: {start+len(dataset.feature_params)+1}")

    # Check which section the leaking features are in
    for idx in feature_indices[:10]:
        idx_val = idx.item()
        if idx_val < target_feat_count:
            print(f"\n⚠️  Feature {idx_val} is in TARGET STATION section!")
        else:
            nearby_idx = (idx_val - target_feat_count) // nearby_feat_count
            pos_in_nearby = (idx_val - target_feat_count) % nearby_feat_count
            if pos_in_nearby == 0:
                feat_type = "distance"
            elif pos_in_nearby <= len(dataset.feature_params):
                feat_type = "weather"
            else:
                feat_type = "SOIL MOISTURE"
            print(f"  Feature {idx_val}: Nearby station {nearby_idx+1}, {feat_type}")

else:
    print(f"\n✓ No leakage detected!")

# Check multiple samples
print(f"\n" + "=" * 70)
print("CHECKING FIRST 10 SAMPLES")
print("=" * 70)
leaky_samples = []
for i in range(min(10, len(dataset))):
    sample = dataset[i]
    features = sample['features']
    target = sample['target'].item()
    matches = (torch.abs(features - target) < 0.0001).sum().item()
    if matches > 0:
        leaky_samples.append(i)
        print(f"Sample {i}: {matches} matches (LEAK!)")
    else:
        print(f"Sample {i}: 0 matches (OK)")

if leaky_samples:
    print(f"\n⚠️  {len(leaky_samples)}/{min(10, len(dataset))} samples have leakage!")
else:
    print(f"\n✓ All checked samples are clean!")

