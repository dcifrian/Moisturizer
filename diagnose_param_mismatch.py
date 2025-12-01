#!/usr/bin/env python3
"""
Diagnose if there's a parameter mismatch between:
1. What's in dense_features.npz
2. What the dataset is requesting
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np

print("="*70)
print("DIAGNOSING PARAMETER MISMATCH")
print("="*70)

# Check what's in dense_features.npz
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'
dense = np.load(dense_path)

print("\n1. WHAT'S IN dense_features.npz:")
print("-" * 70)
dense_params = list(dense['feature_params'])
print(f"Number of parameters: {len(dense_params)}")
print(f"Parameters: {dense_params}")
print()

# Check for soil moisture
soil_moisture_param = 'HS_CV_AVG_-0.2m'
if soil_moisture_param in dense_params:
    soil_idx = dense_params.index(soil_moisture_param)
    print(f"✅ Soil moisture '{soil_moisture_param}' found at index {soil_idx}")
else:
    print(f"❌ Soil moisture '{soil_moisture_param}' NOT in dense_features!")
print()

# Check what the dataset is requesting
collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

print("\n2. WHAT THE DATASET IS REQUESTING:")
print("-" * 70)
print(f"Number of filtered params: {len(filtered_params)}")
print(f"Filtered params: {filtered_params}")
print()

# Check for mismatches
print("\n3. PARAMETER MISMATCH ANALYSIS:")
print("-" * 70)

# Parameters in dense but not in filtered
in_dense_not_filtered = set(dense_params) - set(filtered_params)
if in_dense_not_filtered:
    print(f"In dense_features but NOT in filtered params:")
    for param in in_dense_not_filtered:
        print(f"  - {param}")
else:
    print("✅ All dense params are in filtered params")
print()

# Parameters in filtered but not in dense
in_filtered_not_dense = set(filtered_params) - set(dense_params)
if in_filtered_not_dense:
    print(f"In filtered params but NOT in dense_features:")
    for param in in_filtered_not_dense:
        print(f"  - {param}")
    print()
    print("⚠️  THIS IS THE PROBLEM!")
    print("The dataset expects these parameters, but they're not in dense_features.npz")
    print("This could cause lookups to fail and return -1000.0")
else:
    print("✅ All filtered params are in dense_features")
print()

# Now check what the actual dataset object sees
print("\n4. WHAT THE DATASET OBJECT SEES:")
print("-" * 70)
dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=2,
    n_nearest=5,
    feature_params=filtered_params,
    precomputed_path=None,
    dense_array_path=str(dense_path),
    normalize=False
)

print(f"Dataset feature_params: {dataset.feature_params}")
print(f"Dataset soil_moisture_param: {dataset.soil_moisture_param}")
print()

if hasattr(dataset, 'dense_arrays') and dataset.dense_arrays is not None:
    dataset_dense_params = list(dataset.dense_arrays['feature_params'])
    print(f"Dense array feature_params seen by dataset: {dataset_dense_params}")
    print()

    if dataset.soil_moisture_param in dataset_dense_params:
        idx = dataset_dense_params.index(dataset.soil_moisture_param)
        print(f"✅ Soil moisture in dense features at index {idx}")
    else:
        print(f"❌ Soil moisture NOT in dense features!")
        print(f"   This will cause soil_idx_in_dense to be None")
        print(f"   Which will trigger the fallback to dict lookup")
        print(f"   If dict lookup also fails, target will be -1000.0")
