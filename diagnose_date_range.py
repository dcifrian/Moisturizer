#!/usr/bin/env python3
"""
Check if there's a mismatch between:
1. Date range in raw_timeseries.csv
2. Date range in dense_features.npz
3. Dates the dataset is trying to access
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np
import pandas as pd

print("="*70)
print("DIAGNOSING DATE RANGE MISMATCH")
print("="*70)

# Check date range in dense_features.npz
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'
dense = np.load(dense_path)
dense_dates = pd.DatetimeIndex(dense['dates'])

print("\n1. DATE RANGE IN dense_features.npz:")
print("-" * 70)
print(f"  First date: {dense_dates[0]}")
print(f"  Last date: {dense_dates[-1]}")
print(f"  Total days: {len(dense_dates)}")
print()

# Check date range in raw_timeseries.csv
timeseries_file = Path('./meteogalicia_data') / 'raw_timeseries.csv'
if timeseries_file.exists():
    print("\n2. DATE RANGE IN raw_timeseries.csv:")
    print("-" * 70)
    df = pd.read_csv(timeseries_file)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  First date: {df['date'].min()}")
    print(f"  Last date: {df['date'].max()}")
    print(f"  Unique days: {df['date'].nunique()}")
    print()

    # Check if there are dates in timeseries but not in dense
    timeseries_dates = set(pd.to_datetime(df['date']).dt.normalize())
    dense_dates_set = set(dense_dates.normalize())

    extra_in_timeseries = timeseries_dates - dense_dates_set
    extra_in_dense = dense_dates_set - timeseries_dates

    if extra_in_timeseries:
        print(f"  ⚠️  {len(extra_in_timeseries)} dates in timeseries but NOT in dense_features!")
        print(f"     First few: {sorted(list(extra_in_timeseries))[:5]}")
        print()
    else:
        print("  ✅ All timeseries dates are in dense_features")
        print()

    if extra_in_dense:
        print(f"  ℹ️  {len(extra_in_dense)} dates in dense_features but not in timeseries")
        print()
else:
    print("\n2. raw_timeseries.csv not found")
    print()

# Build dataset and check what dates it's trying to access
print("\n3. DATES THE DATASET IS TRYING TO ACCESS:")
print("-" * 70)

collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

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

if len(dataset.sample_index) > 0:
    # Get all unique dates from sample index
    start_dates = set()
    end_dates = set()
    for sample_info in dataset.sample_index[:100]:  # Check first 100 samples
        start_dates.add(sample_info['start_date'])
        end_dates.add(sample_info['end_date'])

    all_sample_dates = start_dates | end_dates
    print(f"  Sample index has {len(dataset.sample_index)} samples")
    print(f"  First 100 samples use {len(all_sample_dates)} unique dates")
    print(f"  Date range: {min(all_sample_dates)} to {max(all_sample_dates)}")
    print()

    # Check if any sample dates are NOT in dense_features
    dense_dates_set = set(dense_dates.normalize())
    sample_dates_normalized = set(pd.to_datetime(list(all_sample_dates)).normalize())

    missing_dates = sample_dates_normalized - dense_dates_set
    if missing_dates:
        print(f"  ❌ {len(missing_dates)} sample dates NOT in dense_features!")
        print(f"     These lookups will FAIL and return -1000.0")
        print(f"     Missing dates: {sorted(list(missing_dates))[:10]}")
        print()
        print("  This is the problem! The dataset is trying to access dates")
        print("  that don't exist in dense_features.npz.")
    else:
        print("  ✅ All sample dates are in dense_features")
        print()

    # Test a specific lookup
    print("\n4. TESTING SPECIFIC DATE LOOKUP:")
    print("-" * 70)
    first_sample = dataset.sample_index[0]
    print(f"  First sample:")
    print(f"    Station: {first_sample['target_station']}")
    print(f"    Start date: {first_sample['start_date']}")
    print(f"    End date: {first_sample['end_date']}")
    print()

    # Check if this date is in the dense_date_to_idx dictionary
    end_date_normalized = pd.to_datetime(first_sample['end_date']).normalize()
    if end_date_normalized in dataset.dense_date_to_idx:
        idx = dataset.dense_date_to_idx[end_date_normalized]
        print(f"  ✅ End date found in dense_date_to_idx at index {idx}")
    else:
        print(f"  ❌ End date NOT in dense_date_to_idx!")
        print(f"     This will cause fallback to dict method")
        print()
        # Check if it's in the dense_dates at all
        if end_date_normalized in dense_dates_set:
            print(f"     But it IS in dense_dates...")
            print(f"     This means .normalize() fix isn't working!")
        else:
            print(f"     And it's NOT in dense_dates either")
            print(f"     dense_features.npz doesn't have this date")
