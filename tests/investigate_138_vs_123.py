#!/usr/bin/env python3
"""
Investigate the 138 vs 123 feature mismatch reported in map creation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector

collector = MeteoGaliciaCollector()

print("=" * 70)
print("INVESTIGATING 138 vs 123 FEATURE MISMATCH")
print("=" * 70)

# Method 1: buildDataset approach (analyzes timeseries_df directly)
print("\n" + "=" * 70)
print("METHOD 1: buildDataset (direct timeseries analysis)")
print("=" * 70)

timeseries_df = pd.read_csv(collector.timeseries_file)
stations_df = pd.read_csv(collector.stations_file)

soil_moisture_stations = stations_df[stations_df['has_soil_moisture']]['station_id'].tolist()
all_params = timeseries_df['parameter_code'].unique()
soil_moisture_param = "HS_CV_AVG_-0.2m"
coverage_threshold = 0.25

print(f"\nAnalyzing {len(all_params)} parameters on {len(soil_moisture_stations)} soil moisture stations")
print(f"Coverage threshold: {coverage_threshold * 100:.0f}%\n")

filtered_params_method1 = []
for param in sorted(all_params):
    if param == soil_moisture_param:
        continue

    param_data = timeseries_df[
        (timeseries_df['parameter_code'] == param) &
        (timeseries_df['station_id'].isin(soil_moisture_stations))
    ]

    stations_with_param = param_data['station_id'].nunique()
    coverage = stations_with_param / len(soil_moisture_stations) if soil_moisture_stations else 0

    status = "✓" if coverage >= coverage_threshold else "✗"
    print(f"{status} {param:30s}: {coverage*100:5.1f}% ({stations_with_param}/{len(soil_moisture_stations)} stations)")

    if coverage >= coverage_threshold:
        filtered_params_method1.append(param)

print(f"\nMethod 1 result: {len(filtered_params_method1)} parameters")
print(f"  → Total features: {len(filtered_params_method1)} + 4*(1 + {len(filtered_params_method1)} + 1) = {len(filtered_params_method1) + 4*(1 + len(filtered_params_method1) + 1)}")

# Method 2: analyze_parameter_coverage (from ml_ready_dataset.csv)
print("\n" + "=" * 70)
print("METHOD 2: analyze_parameter_coverage (ml_ready_dataset.csv)")
print("=" * 70)

ml_dataset_file = collector.data_dir / "ml_ready_dataset.csv"

if ml_dataset_file.exists():
    print(f"\n✓ Found ml_ready_dataset.csv")
    print(f"  Analyzing parameter coverage...")

    try:
        coverage_dict, filtered_params_method2 = collector.analyze_parameter_coverage(
            coverage_threshold=coverage_threshold
        )
        print(f"\nMethod 2 result: {len(filtered_params_method2)} parameters")
        print(f"  → Total features: {len(filtered_params_method2)} + 4*(1 + {len(filtered_params_method2)} + 1) = {len(filtered_params_method2) + 4*(1 + len(filtered_params_method2) + 1)}")

        # Compare the two methods
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)

        set1 = set(filtered_params_method1)
        set2 = set(filtered_params_method2)

        common = sorted(set1 & set2)
        only_method1 = sorted(set1 - set2)
        only_method2 = sorted(set2 - set1)

        print(f"\nCommon parameters: {len(common)}")
        if len(common) <= 30:
            for p in common:
                print(f"  - {p}")

        if only_method1:
            print(f"\nOnly in Method 1 (timeseries, NOT in ml_ready_dataset.csv): {len(only_method1)}")
            for p in only_method1:
                print(f"  - {p}")

        if only_method2:
            print(f"\nOnly in Method 2 (ml_ready_dataset.csv, NOT in timeseries): {len(only_method2)}")
            for p in only_method2:
                print(f"  - {p}")

        if len(filtered_params_method2) == 26:
            print(f"\n⚠️  Method 2 has 26 params → 138 total features (matches your error!)")
        if len(filtered_params_method1) == 23:
            print(f"⚠️  Method 1 has 23 params → 123 total features (matches precomputed data!)")

        print(f"\n" + "=" * 70)
        print(f"CONCLUSION")
        print(f"=" * 70)

        if only_method1 or only_method2:
            print(f"\n❌ The two methods produce DIFFERENT parameter lists!")
            print(f"   This explains the 138 vs 123 feature mismatch:")
            print(f"   - Training uses Method 1 → 123 features")
            print(f"   - Map creation uses Method 2 → 138 features")
            print(f"\n   The difference is {len(only_method2)} extra parameters in ml_ready_dataset.csv")
            print(f"   and {len(only_method1)} missing from ml_ready_dataset.csv")
        else:
            print(f"\n✓ Both methods produce the SAME parameter lists")
            print(f"   The feature count should match.")

    except Exception as e:
        print(f"\n✗ Error analyzing ml_ready_dataset.csv: {e}")
        filtered_params_method2 = None

else:
    print(f"\n✗ ml_ready_dataset.csv not found at {ml_dataset_file}")
    print(f"   Map creation would fail when trying to call analyze_parameter_coverage()")
    print(f"\n   To fix this, you need to either:")
    print(f"   1. Generate ml_ready_dataset.csv by calling collector.create_ml_ready_dataset()")
    print(f"   2. Pass the same filtered_params used during training to map creation")

# Check what was actually used to build the precomputed data
print("\n" + "=" * 70)
print("WHAT WAS ACTUALLY USED FOR PRECOMPUTED DATA")
print("=" * 70)

precomp_path = collector.data_dir / "precomputed_sequences.npz"
if precomp_path.exists():
    data = np.load(precomp_path)
    print(f"\nPrecomputed data shape: {data['features'].shape}")
    print(f"  [num_samples, seq_length, total_features]")
    print(f"  Total features per timestep: {data['features'].shape[2]}")

    if data['features'].shape[2] == 123:
        print(f"\n  → Precomputed with 23 parameters (Method 1)")
    elif data['features'].shape[2] == 138:
        print(f"\n  → Precomputed with 26 parameters (Method 2)")
else:
    print(f"\n✗ Precomputed data not found")
