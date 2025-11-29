#!/usr/bin/env python3
"""
Diagnose feature mismatch between different code paths
"""

import numpy as np
import pandas as pd
from Moisturizer import MeteoGaliciaCollector

collector = MeteoGaliciaCollector()

print("=" * 70)
print("DIAGNOSING FEATURE SELECTION MISMATCH")
print("=" * 70)

# Method 1: analyze_parameter_coverage (uses ml_ready_dataset.csv)
print("\n" + "=" * 70)
print("METHOD 1: analyze_parameter_coverage (from ml_ready_dataset.csv)")
print("=" * 70)

try:
    coverage_dict, filtered_params_csv = collector.analyze_parameter_coverage(
        coverage_threshold=0.25
    )
    print(f"\n✓ Found {len(filtered_params_csv)} parameters from ml_ready_dataset.csv")
    print(f"Parameters: {sorted(filtered_params_csv)}")
except Exception as e:
    print(f"✗ Error: {e}")
    filtered_params_csv = None

# Method 2: Direct timeseries analysis (like buildDataset does)
print("\n" + "=" * 70)
print("METHOD 2: Direct timeseries analysis (like buildDataset)")
print("=" * 70)

# Load timeseries and stations
timeseries_df = pd.read_csv(collector.timeseries_file)
stations_df = pd.read_csv(collector.stations_file)

soil_moisture_stations = stations_df[stations_df['has_soil_moisture']]['station_id'].tolist()
all_params = timeseries_df['parameter_code'].unique()
soil_moisture_param = "HS_CV_AVG_-0.2m"
coverage_threshold = 0.25
filtered_params_timeseries = []

print(f"\nAnalyzing {len(all_params)} parameters on {len(soil_moisture_stations)} stations with soil moisture...")
print(f"Coverage threshold: {coverage_threshold * 100:.0f}%\n")

for param in sorted(all_params):
    if param == soil_moisture_param:
        continue  # Skip soil moisture - it's the target

    # Count how many soil moisture stations have this parameter
    param_data = timeseries_df[
        (timeseries_df['parameter_code'] == param) &
        (timeseries_df['station_id'].isin(soil_moisture_stations))
    ]

    stations_with_param = param_data['station_id'].nunique()
    coverage = stations_with_param / len(soil_moisture_stations) if soil_moisture_stations else 0

    status = "✓" if coverage >= coverage_threshold else "✗"
    print(f"{status} {param:30s}: {coverage*100:5.1f}% ({stations_with_param}/{len(soil_moisture_stations)} stations)")

    if coverage >= coverage_threshold:
        filtered_params_timeseries.append(param)

print(f"\n✓ Found {len(filtered_params_timeseries)} parameters from direct timeseries analysis")
print(f"Parameters: {sorted(filtered_params_timeseries)}")

# Compare the two methods
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

if filtered_params_csv is not None:
    set_csv = set(filtered_params_csv)
    set_timeseries = set(filtered_params_timeseries)

    only_csv = sorted(set_csv - set_timeseries)
    only_timeseries = sorted(set_timeseries - set_csv)
    common = sorted(set_csv & set_timeseries)

    print(f"\nCommon parameters: {len(common)}")
    print(f"  {common}")

    if only_csv:
        print(f"\nOnly in CSV method: {len(only_csv)}")
        print(f"  {only_csv}")

    if only_timeseries:
        print(f"\nOnly in timeseries method: {len(only_timeseries)}")
        print(f"  {only_timeseries}")

    if not only_csv and not only_timeseries:
        print(f"\n✓ Both methods produce IDENTICAL parameter lists!")
    else:
        print(f"\n⚠️  Methods produce DIFFERENT parameter lists!")
        print(f"   This could explain the feature mismatch (138 vs 123)")

# Check dense arrays
print("\n" + "=" * 70)
print("DENSE ARRAY ANALYSIS")
print("=" * 70)

dense_array_path = collector.data_dir / "dense_features.npz"
if dense_array_path.exists():
    dense_data = np.load(dense_array_path)
    dense_feature_params = dense_data['feature_params'].tolist()

    print(f"\nDense array feature_params: {len(dense_feature_params)}")
    print(f"  {sorted(dense_feature_params)}")

    if soil_moisture_param in dense_feature_params:
        soil_idx = dense_feature_params.index(soil_moisture_param)
        print(f"\n⚠️  Soil moisture IS in dense array at index {soil_idx} (out of {len(dense_feature_params)-1})")
        print(f"  This is EXPECTED - soil moisture should be the LAST feature in dense array")
        print(f"  It should NOT be copied into target station features")
    else:
        print(f"\n✗ Soil moisture NOT in dense array (unexpected!)")

    # Compare dense array params with filtered params
    dense_without_soil = [p for p in dense_feature_params if p != soil_moisture_param]

    if filtered_params_csv is not None:
        if set(dense_without_soil) == set(filtered_params_csv):
            print(f"\n✓ Dense array params (without soil) match CSV filtered params")
        else:
            print(f"\n⚠️  Dense array params (without soil) DON'T match CSV filtered params")

    if set(dense_without_soil) == set(filtered_params_timeseries):
        print(f"✓ Dense array params (without soil) match timeseries filtered params")
    else:
        print(f"⚠️  Dense array params (without soil) DON'T match timeseries filtered params")
        only_dense = sorted(set(dense_without_soil) - set(filtered_params_timeseries))
        only_filtered = sorted(set(filtered_params_timeseries) - set(dense_without_soil))
        if only_dense:
            print(f"   Only in dense: {only_dense}")
        if only_filtered:
            print(f"   Only in filtered: {only_filtered}")

else:
    print(f"\n✗ Dense array not found at {dense_array_path}")

# Check for actual leakage in sequence building
print("\n" + "=" * 70)
print("CHECKING SEQUENCE BUILDING FOR LEAKAGE")
print("=" * 70)

try:
    from Moisturizer import SoilMoistureSequenceDataset

    # Load dataset like training does
    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=4,
        feature_params=filtered_params_timeseries,
        dense_array_path=str(dense_array_path)
    )

    print(f"\nDataset created with {len(dataset.feature_params)} feature_params")
    print(f"  Feature params: {sorted(dataset.feature_params)}")

    if soil_moisture_param in dataset.feature_params:
        print(f"\n⚠️  LEAKAGE SOURCE FOUND:")
        print(f"  Soil moisture IS in dataset.feature_params!")
        print(f"  This means it will be copied into target station features at line 1229")
    else:
        print(f"\n✓ dataset.feature_params does NOT include soil moisture (good!)")

    # Check dense arrays
    if dataset.dense_arrays is not None:
        dense_feat_params = dataset.dense_arrays['feature_params']
        print(f"\nDataset dense_arrays['feature_params']: {len(dense_feat_params)}")
        print(f"  {sorted(dense_feat_params)}")

        if soil_moisture_param in dense_feat_params:
            soil_idx_dense = dense_feat_params.index(soil_moisture_param)
            print(f"\n  Soil moisture at index {soil_idx_dense} in dense array")

        # The key check: target_features_per_timestep
        target_features_per_timestep = len(dataset.feature_params)
        print(f"\ntarget_features_per_timestep = len(dataset.feature_params) = {target_features_per_timestep}")
        print(f"This is how many features we copy from dense array at line 1229")

        if soil_idx_dense < target_features_per_timestep:
            print(f"\n❌ LEAKAGE BUG FOUND!")
            print(f"  Soil moisture is at index {soil_idx_dense} in dense array")
            print(f"  We copy indices 0 to {target_features_per_timestep-1} from dense array")
            print(f"  This INCLUDES soil moisture! (index {soil_idx_dense} < {target_features_per_timestep})")
        else:
            print(f"\n✓ No leakage from this path")
            print(f"  Soil moisture at index {soil_idx_dense} >= {target_features_per_timestep}")
            print(f"  We only copy indices 0-{target_features_per_timestep-1}, so soil is excluded")

except Exception as e:
    print(f"\n✗ Error loading dataset: {e}")
    import traceback
    traceback.print_exc()
