#!/usr/bin/env python3
"""
Trace the EXACT lookup for the first sample to see what's happening.
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np
import pandas as pd

print("="*70)
print("TRACING EXACT LOOKUP FOR FIRST SAMPLE")
print("="*70)

# Build dataset
collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'

print("\nBuilding dataset...")
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

print(f"Dataset has {len(dataset)} samples")
print()

# Get first sample info
first_sample_info = dataset.sample_index[0]
print("First sample info from sample_index:")
print(f"  target_station: {first_sample_info['target_station']}")
print(f"  start_date: {first_sample_info['start_date']}")
print(f"  end_date: {first_sample_info['end_date']}")
print()

# Load dense arrays directly
dense = np.load(dense_path)
dense_dates = pd.DatetimeIndex(dense['dates'])
dense_station_ids = dense['station_ids']
dense_features = dense['features']
dense_params = list(dense['feature_params'])

print("Dense arrays info:")
print(f"  Shape: {dense_features.shape}")
print(f"  Dates: {len(dense_dates)} from {dense_dates[0]} to {dense_dates[-1]}")
print(f"  Stations: {len(dense_station_ids)}")
print(f"  Parameters: {len(dense_params)}")
print()

# Find soil moisture index
soil_param = 'HS_CV_AVG_-0.2m'
soil_idx_in_dense = dense_params.index(soil_param)
print(f"Soil moisture '{soil_param}' at index {soil_idx_in_dense}")
print()

# Manually perform the lookup
target_station_id = first_sample_info['target_station']
end_date = pd.to_datetime(first_sample_info['end_date'])

print("Manual lookup:")
print(f"  Looking up station {target_station_id} on date {end_date}")
print()

# Find station index
if target_station_id in dense_station_ids:
    target_idx = np.where(dense_station_ids == target_station_id)[0][0]
    print(f"  ✅ Station {target_station_id} found at index {target_idx}")
else:
    print(f"  ❌ Station {target_station_id} NOT in dense arrays!")
    target_idx = None

# Find date index (with normalize)
end_date_normalized = end_date.normalize()
if end_date_normalized in dense_dates.normalize():
    # Find which index
    date_indices = np.where(dense_dates.normalize() == end_date_normalized)[0]
    if len(date_indices) > 0:
        end_date_idx = date_indices[0]
        print(f"  ✅ Date {end_date_normalized} found at index {end_date_idx}")
        print(f"     Dense date at that index: {dense_dates[end_date_idx]}")
    else:
        print(f"  ❌ Date {end_date_normalized} NOT found!")
        end_date_idx = None
else:
    print(f"  ❌ Date {end_date_normalized} NOT in dense dates!")
    end_date_idx = None

print()

# If we have both indices, look up the value
if target_idx is not None and end_date_idx is not None:
    print("Looking up soil moisture value:")
    print(f"  dense_features[{target_idx}, {end_date_idx}, {soil_idx_in_dense}]")

    target_value = dense_features[target_idx, end_date_idx, soil_idx_in_dense]
    print(f"  Value: {target_value}")
    print()

    if target_value == -1000.0 or target_value == -9999.0:
        print("  ❌ VALUE IS INVALID MARKER!")
        print()
        print("  DIAGNOSIS:")
        print("  The dense_features.npz file contains -1000.0 at this exact location.")
        print("  This means the soil moisture data is missing for this station/date.")
        print()

        # Check the entire column for this station
        all_sm_values = dense_features[target_idx, :, soil_idx_in_dense]
        valid_sm = all_sm_values[(all_sm_values != -1000.0) & (all_sm_values != -9999.0)]
        print(f"  Station {target_station_id} soil moisture stats:")
        print(f"    Total dates: {len(all_sm_values)}")
        print(f"    Valid values: {len(valid_sm)} ({100*len(valid_sm)/len(all_sm_values):.1f}%)")
        if len(valid_sm) > 0:
            print(f"    Valid range: [{valid_sm.min():.3f}, {valid_sm.max():.3f}]")
        print()

        # Check why this sample is in sample_index if target is invalid
        print("  ❓ WHY IS THIS SAMPLE IN sample_index?")
        print("  The sample_index builder should have filtered out samples with invalid targets.")
        print()
        print("  Checking timeseries_df directly...")
        timeseries_df = dataset.timeseries_df
        soil_df = timeseries_df[
            (timeseries_df['station_id'] == target_station_id) &
            (timeseries_df['parameter_code'] == soil_param) &
            (timeseries_df['date'] == end_date)
        ]
        if len(soil_df) > 0:
            csv_value = soil_df.iloc[0]['value']
            print(f"  Value in raw_timeseries.csv: {csv_value}")
            print()
            if csv_value != target_value:
                print("  ❌ MISMATCH FOUND!")
                print(f"     raw_timeseries.csv has: {csv_value}")
                print(f"     dense_features.npz has: {target_value}")
                print()
                print("  DIAGNOSIS:")
                print("  dense_features.npz is OUTDATED!")
                print("  It doesn't match the data in raw_timeseries.csv")
                print("  You need to rebuild dense_features.npz")
        else:
            print("  ❌ No data in raw_timeseries.csv for this station/date/parameter!")
            print("     This is a bug in the sample_index builder.")
    else:
        print("  ✅ VALUE IS VALID!")
        print()
        print("  This is confusing - the lookup returns a valid value,")
        print("  but the dataset returns -1000.0. There must be a bug")
        print("  in how _build_sequence_from_dense extracts the target.")

print()

# Now get the sample from the dataset
print("="*70)
print("GETTING SAMPLE FROM DATASET")
print("="*70)
sample = dataset[0]
dataset_target = sample['target'].numpy()[0]
print(f"Dataset returned target: {dataset_target}")
print()

if target_idx is not None and end_date_idx is not None:
    manual_target = dense_features[target_idx, end_date_idx, soil_idx_in_dense]
    if dataset_target != manual_target:
        print(f"❌ MISMATCH!")
        print(f"   Manual lookup: {manual_target}")
        print(f"   Dataset returned: {dataset_target}")
        print()
        print("   This means there's a bug in _build_sequence_from_dense")
    else:
        print(f"✅ Match!")
        print(f"   Both return: {dataset_target}")
