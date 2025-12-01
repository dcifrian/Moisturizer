#!/usr/bin/env python3
"""
Debug why the dataset only selects samples with invalid targets (-1000.0)
even though dense_features.npz has 98.4% valid soil moisture data
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np

print("="*70)
print("DEBUGGING SAMPLE SELECTION LOGIC")
print("="*70)

# First, check what's in dense_features directly
dense = np.load('./meteogalicia_data/dense_features.npz')
soil_moisture = dense['features'][:, :, -1]  # Last column
dates = dense['dates']

print(f"\nDense features:")
print(f"  Stations: {soil_moisture.shape[0]}")
print(f"  Dates: {soil_moisture.shape[1]}")
print(f"  First date: {dates[0]}")
print(f"  Last date: {dates[-1]}")
print()

# Check a specific station's soil moisture
station_0_sm = soil_moisture[0, :]
valid_sm_0 = station_0_sm[station_0_sm > 0]  # Valid values
print(f"Station 0 soil moisture:")
print(f"  Valid values: {len(valid_sm_0)}/{len(station_0_sm)}")
if len(valid_sm_0) > 0:
    print(f"  Range: [{valid_sm_0.min():.3f}, {valid_sm_0.max():.3f}]")
    print(f"  First 10 valid: {valid_sm_0[:10]}")
print()

# Now load dataset and see what it selects
collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'

print("Loading dataset with seq_length=2...")
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

print(f"\nDataset built {len(base_dataset.sample_index)} samples")
print()

# Check what indices the dataset selected
if len(base_dataset.sample_index) > 0:
    first_sample_info = base_dataset.sample_index[0]
    print(f"First sample in index:")
    print(f"  Station: {first_sample_info['target_station']}")
    print(f"  End date: {first_sample_info['end_date']}")
    print(f"  Start date: {first_sample_info['start_date']}")
    print()

    # Try to manually fetch the soil moisture for this sample
    station_id = first_sample_info['target_station']
    end_date = first_sample_info['end_date']

    # Find which row in dense_features corresponds to this
    station_ids = dense['station_ids']
    if station_id in station_ids:
        station_idx = list(station_ids).index(station_id)
        print(f"  Station index in dense_features: {station_idx}")

        # Find date index
        date_idx = None
        for i, d in enumerate(dates):
            if d == end_date:
                date_idx = i
                break

        if date_idx is not None:
            print(f"  Date index in dense_features: {date_idx}")
            sm_value = soil_moisture[station_idx, date_idx]
            print(f"  Soil moisture in dense_features: {sm_value}")
            print()
        else:
            print(f"  ❌ Date {end_date} not found in dense dates!")
            print(f"  Dense dates range: {dates[0]} to {dates[-1]}")
            print()

    # Now get what the dataset actually returns
    print(f"Getting sample from dataset...")
    sample = base_dataset[0]
    target = sample['target'].numpy()
    print(f"  Target returned by dataset: {target}")
    print()

    if target[0] == -1000.0:
        print("  ❌ Dataset is returning invalid target even though dense_features has valid data!")
        print("  This suggests the dataset is looking at the wrong date or wrong station.")
    else:
        print("  ✅ Dataset returned valid target")
