#!/usr/bin/env python3
"""
Complete diagnostic to identify why all targets are -1000.0
This runs all checks and provides a clear diagnosis.
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np
import pandas as pd

print("="*70)
print("COMPLETE DIAGNOSTIC FOR -1000.0 TARGET ISSUE")
print("="*70)

# Load dense features
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'
dense = np.load(dense_path)
dense_dates = pd.DatetimeIndex(dense['dates'])

print("\n" + "="*70)
print("1. CHECKING DENSE_FEATURES.NPZ DATA QUALITY")
print("="*70)

# Check soil moisture in dense_features
soil_moisture = dense['features'][:, :, -1]  # Last column
valid_sm = soil_moisture[(soil_moisture != -1000.0) & (soil_moisture != -9999.0)]

print(f"Soil moisture in dense_features.npz:")
print(f"  Total values: {soil_moisture.size:,}")
print(f"  Valid values: {len(valid_sm):,} ({100*len(valid_sm)/soil_moisture.size:.1f}%)")
if len(valid_sm) > 0:
    print(f"  Valid range: [{valid_sm.min():.3f}, {valid_sm.max():.3f}]")
print(f"  Date range: {dense_dates[0]} to {dense_dates[-1]} ({len(dense_dates)} days)")
print()

print("\n" + "="*70)
print("2. BUILDING BASE DATASET (AS PRECOMPUTE_AUGMENTED.PY DOES)")
print("="*70)

# Build dataset exactly as precompute_augmented.py does
collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

base_dataset = SoilMoistureSequenceDataset(
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

print(f"\nDataset has {len(base_dataset)} samples")
print()

print("\n" + "="*70)
print("3. CHECKING SAMPLE DATES VS DENSE_FEATURES DATES")
print("="*70)

# Check what dates the sample_index contains
if len(base_dataset.sample_index) > 0:
    sample_dates = set()
    for sample_info in base_dataset.sample_index[:min(100, len(base_dataset.sample_index))]:
        sample_dates.add(pd.to_datetime(sample_info['start_date']).normalize())
        sample_dates.add(pd.to_datetime(sample_info['end_date']).normalize())

    dense_dates_set = set(dense_dates.normalize())

    print(f"First 100 samples use {len(sample_dates)} unique dates")
    print(f"dense_features.npz has {len(dense_dates_set)} unique dates")
    print()

    # Check for missing dates
    missing_in_dense = sample_dates - dense_dates_set
    if missing_in_dense:
        print(f"❌ FOUND THE PROBLEM!")
        print(f"   {len(missing_in_dense)} sample dates are NOT in dense_features.npz!")
        print(f"   Sample dates range: {min(sample_dates)} to {max(sample_dates)}")
        print(f"   Dense dates range: {min(dense_dates_set)} to {max(dense_dates_set)}")
        print()
        print(f"   Missing dates (first 10): {sorted(list(missing_in_dense))[:10]}")
        print()
        print("   DIAGNOSIS:")
        print("   - dense_features.npz is OUTDATED or INCOMPLETE")
        print("   - raw_timeseries.csv has data that dense_features.npz doesn't have")
        print("   - When the dataset tries to access these dates, lookups fail")
        print("   - This causes all targets to return -1000.0")
        print()
        print("   SOLUTION:")
        print("   - Rebuild dense_features.npz using buildDataset() with your full data")
        print("   - Or use precomputed dataset instead of dense_features")
    else:
        print("✅ All sample dates exist in dense_features.npz")
        print()

print("\n" + "="*70)
print("4. TESTING ACTUAL SAMPLE RETRIEVAL")
print("="*70)

# Test first 10 samples
if len(base_dataset) > 0:
    print("Testing first 10 samples...")
    targets = []
    for i in range(min(10, len(base_dataset))):
        sample = base_dataset[i]
        targets.append(sample['target'].numpy()[0])

    targets = np.array(targets)
    print(f"  Targets: {targets}")
    print(f"  Range: [{targets.min():.6f}, {targets.max():.6f}]")
    print()

    invalid_count = np.sum((targets == -1000.0) | (targets == -9999.0))
    if invalid_count == len(targets):
        print("❌ ALL TARGETS ARE -1000.0!")
        print()
        if len(missing_in_dense) > 0:
            print("   This confirms the date mismatch issue above.")
        else:
            print("   This is unexpected - dates exist but targets are still invalid.")
            print("   Run check_base_dataset.py with DEBUG logging to trace the issue.")
    elif invalid_count > 0:
        print(f"⚠️  {invalid_count}/{len(targets)} targets are invalid")
    else:
        print("✅ All targets are valid!")

print("\n" + "="*70)
print("5. CHECKING PARAMETER CONFIGURATION")
print("="*70)

soil_param = 'HS_CV_AVG_-0.2m'
dense_params = list(dense['feature_params'])

if soil_param in dense_params:
    soil_idx = dense_params.index(soil_param)
    print(f"✅ Soil moisture '{soil_param}' found at index {soil_idx} in dense_features")
else:
    print(f"❌ Soil moisture '{soil_param}' NOT in dense_features!")
    print(f"   Dense params: {dense_params}")
    print()
    print("   DIAGNOSIS:")
    print("   - dense_features.npz was created without soil moisture parameter")
    print("   - All target lookups will fail")
    print()
    print("   SOLUTION:")
    print("   - Rebuild dense_features.npz with correct parameters")

print()
print("="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
