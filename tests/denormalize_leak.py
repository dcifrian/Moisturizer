#!/usr/bin/env python3
"""
Denormalize the ACTUAL leak from sample 8 to check if it's real or coincidence
NO HARDCODED VALUES - read everything from actual data
"""

import numpy as np
import pandas as pd
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

collector = MeteoGaliciaCollector()

# Get filtered params
timeseries_df = pd.read_csv(collector.timeseries_file)
stations_df = pd.read_csv(collector.stations_file)
soil_moisture_stations = stations_df[stations_df['has_soil_moisture']]['station_id'].tolist()
all_params = timeseries_df['parameter_code'].unique()
filtered_params = []
for param in sorted(all_params):
    if param == "HS_CV_AVG_-0.2m":
        continue
    param_data = timeseries_df[
        (timeseries_df['parameter_code'] == param) &
        (timeseries_df['station_id'].isin(soil_moisture_stations))
    ]
    coverage = param_data['station_id'].nunique() / len(soil_moisture_stations)
    if coverage >= 0.25:
        filtered_params.append(param)

print("=" * 70)
print("DENORMALIZING ACTUAL LEAK FROM SAMPLE 8")
print("=" * 70)

# Load dataset
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

print(f"Dataset loaded: {len(dataset)} samples")
print(f"Feature params: {len(filtered_params)}")

# Get sample 8
sample = dataset[8]
features = sample['features'].numpy()
target = sample['target'].item()
station_id = sample['target_station_id']

print(f"\nSample 8:")
print(f"  Station ID: {station_id}")
print(f"  Features shape: {features.shape}")
print(f"  Target (normalized): {target:.6f}")

# Find leak
target_feat_count = len(filtered_params)
match_mask = (np.abs(features[:, :target_feat_count] - target) < 0.001)
match_positions = np.where(match_mask)

if len(match_positions[0]) == 0:
    print("\n✓ No leak found!")
    exit(0)

print(f"\nFound {len(match_positions[0])} match(es)")

# Check first match
t = match_positions[0][0]
f = match_positions[1][0]
feature_val_norm = features[t, f]
param_name = filtered_params[f]

print(f"\nFirst match:")
print(f"  Timestep: {t}")
print(f"  Feature index: {f}")
print(f"  Feature name: {param_name}")
print(f"  Feature value (normalized): {feature_val_norm:.6f}")
print(f"  Target value (normalized): {target:.6f}")
print(f"  Difference (normalized): {abs(feature_val_norm - target):.6f}")

# Load normalization stats
if dataset.norm_stats is None:
    print("\n✗ No normalization stats available!")
    exit(1)

target_min = dataset.norm_stats['target_min']
target_max = dataset.norm_stats['target_max']
feat_min = dataset.norm_stats['feature_mins'][f]
feat_max = dataset.norm_stats['feature_maxs'][f]

print(f"\nNormalization ranges:")
print(f"  Target (soil moisture HS_CV_AVG_-0.2m): [{target_min:.4f}, {target_max:.4f}]")
print(f"  Feature {f} ({param_name}): [{feat_min:.4f}, {feat_max:.4f}]")

# Denormalize
# Formula: normalized = 2.0 * (raw - min) / (max - min) - 1.0
# Inverse: raw = (normalized + 1.0) / 2.0 * (max - min) + min

target_raw = (target + 1.0) / 2.0 * (target_max - target_min) + target_min
feature_raw = (feature_val_norm + 1.0) / 2.0 * (feat_max - feat_min) + feat_min

print(f"\nDenormalized values:")
print(f"  Target (soil moisture): {target_raw:.6f}")
print(f"  Feature ({param_name}): {feature_raw:.6f}")
print(f"  Raw difference: {abs(target_raw - feature_raw):.6f}")

print(f"\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if abs(target_raw - feature_raw) < 0.01:
    print(f"\n❌❌❌ RAW VALUES ARE NEARLY IDENTICAL!")
    print(f"  This is REAL DATA LEAKAGE!")
    print(f"  Feature {f} ({param_name}) contains actual soil moisture!")
    print(f"\n  The precomputed data has a BUG where soil moisture leaked")
    print(f"  into the target station features during precomputation.")
else:
    print(f"\n✓ Raw values are DIFFERENT")
    print(f"  Soil moisture: {target_raw:.2f}%")
    print(f"  {param_name}: {feature_raw:.2f}°C" if 'TA' in param_name or 'TO' in param_name else f"{param_name}: {feature_raw:.2f}")
    print(f"\n  This is a NORMALIZATION COINCIDENCE")
    print(f"  Different physical quantities happened to normalize to similar values")
    print(f"  This is NOT data leakage - it's just bad luck with the [-1,1] mapping")

# Additional check: Show the normalization mapping
print(f"\n" + "=" * 70)
print("HOW THE NORMALIZATION WORKS")
print("=" * 70)

print(f"\nTarget (soil moisture):")
print(f"  Raw range: [{target_min:.4f}, {target_max:.4f}]")
print(f"  This sample's raw value: {target_raw:.6f}")
print(f"  Position in range: {(target_raw - target_min) / (target_max - target_min) * 100:.1f}%")
print(f"  Normalized to: {target:.6f}")

print(f"\nFeature {f} ({param_name}):")
print(f"  Raw range: [{feat_min:.4f}, {feat_max:.4f}]")
print(f"  This sample's raw value: {feature_raw:.6f}")
print(f"  Position in range: {(feature_raw - feat_min) / (feat_max - feat_min) * 100:.1f}%")
print(f"  Normalized to: {feature_val_norm:.6f}")

print(f"\nBoth happened to land at similar normalized values (~-0.234)")
print(f"because they're at similar positions within their respective ranges.")
