#!/usr/bin/env python3
"""
Build dense feature array for ultra-fast sequence generation

Instead of building sequences one-by-one with massive redundancy,
pre-compute a [stations × dates × features] array ONCE, then slice it.

Speedup: From 30 min → ~2-3 minutes for the preprocessing, then instant sequence building
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
from Moisturizer import MeteoGaliciaCollector

def build_dense_feature_array(
    timeseries_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    feature_params: List[str],
    soil_moisture_param: str = "HS_CV_AVG_-0.2m",
    missing_value: float = -1000.0
) -> Tuple[np.ndarray, np.ndarray, List[int], pd.DatetimeIndex]:
    """
    Build dense feature array for all stations × all dates × all parameters

    IMPORTANT: Includes soil moisture in the array! This is needed for nearby stations.
    (Target stations will exclude it during sequence building to prevent leakage)

    Args:
        timeseries_df: Raw timeseries data
        stations_df: Station metadata
        feature_params: List of parameter codes to include (WITHOUT soil moisture)
        soil_moisture_param: Soil moisture parameter to add separately
        missing_value: Value for missing data

    Returns:
        features_array: [num_stations, num_dates, num_features] array
        mask_array: [num_stations, num_dates, num_features] mask (1=valid, 0=missing)
        station_ids: List of station IDs (index matches array)
        date_index: DatetimeIndex of all dates
    """
    print("=" * 70)
    print("BUILDING DENSE FEATURE ARRAY (ONE-TIME PREPROCESSING)")
    print("=" * 70)

    # Get all unique dates and stations
    all_dates = sorted(timeseries_df['date'].unique())
    date_index = pd.DatetimeIndex(all_dates)
    station_ids = sorted(stations_df['station_id'].unique())

    # Build combined parameter list: features + soil moisture
    all_params = feature_params + [soil_moisture_param]

    num_stations = len(station_ids)
    num_dates = len(date_index)
    num_features = len(all_params)

    print(f"\nArray dimensions:")
    print(f"  Stations: {num_stations}")
    print(f"  Dates: {num_dates}")
    print(f"  Features: {len(feature_params)} weather params + 1 soil moisture = {num_features} total")
    print(f"  Total elements: {num_stations * num_dates * num_features:,}")
    print(f"  Memory: ~{num_stations * num_dates * num_features * 4 / 1e6:.1f} MB")

    # Initialize arrays
    features_array = np.full((num_stations, num_dates, num_features), missing_value, dtype=np.float32)
    mask_array = np.zeros((num_stations, num_dates, num_features), dtype=np.float32)

    # Create mapping for fast indexing
    station_to_idx = {sid: idx for idx, sid in enumerate(station_ids)}
    date_to_idx = {date: idx for idx, date in enumerate(date_index)}
    param_to_idx = {param: idx for idx, param in enumerate(all_params)}

    print("\nFilling array with data...")
    # Fill the array - vectorized operation on grouped data
    for param_idx, param in enumerate(all_params):
        if param_idx % 10 == 0:
            print(f"  Processing parameter {param_idx+1}/{num_features}...")

        # Get all data for this parameter at once
        param_data = timeseries_df[timeseries_df['parameter_code'] == param]

        # Fill in bulk using numpy indexing
        for _, row in param_data.iterrows():
            station_idx = station_to_idx.get(row['station_id'])
            date_idx = date_to_idx.get(row['date'])

            if station_idx is not None and date_idx is not None:
                features_array[station_idx, date_idx, param_idx] = row['value']
                mask_array[station_idx, date_idx, param_idx] = 1.0

    print(f"\n✓ Dense array built!")
    print(f"  Valid data points: {mask_array.sum():,.0f}")
    print(f"  Missing data points: {(mask_array == 0).sum():,.0f}")
    print(f"  Coverage: {mask_array.sum() / mask_array.size * 100:.1f}%")

    return features_array, mask_array, station_ids, date_index


def save_dense_arrays(
    output_path: str,
    features_array: np.ndarray,
    mask_array: np.ndarray,
    station_ids: List[int],
    date_index: pd.DatetimeIndex,
    all_params: List[str]
):
    """Save dense arrays to disk"""
    print(f"\nSaving dense arrays to {output_path}...")

    np.savez_compressed(
        output_path,
        features=features_array,
        masks=mask_array,
        station_ids=np.array(station_ids, dtype=np.int32),
        dates=date_index.values.astype('datetime64[ns]').astype(np.int64),  # Unix timestamps
        feature_params=np.array(all_params, dtype='U50')  # Unicode strings (includes soil moisture!)
    )

    print(f"✓ Saved!")
    print(f"  File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")


def main():
    print("Building dense feature array for fast sequence generation...")

    # Load data
    collector = MeteoGaliciaCollector()

    print("\nLoading timeseries data...")
    timeseries_df = pd.read_csv(collector.timeseries_file)
    timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])

    print("Loading station metadata...")
    stations_df = pd.read_csv(collector.stations_file)

    # Get feature parameters
    print("\nAnalyzing parameter coverage...")
    coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

    print(f"\n✓ Selected {len(filtered_params)} parameters")

    # Build dense arrays (includes soil moisture for nearby stations!)
    features_array, mask_array, station_ids, date_index = build_dense_feature_array(
        timeseries_df=timeseries_df,
        stations_df=stations_df,
        feature_params=filtered_params,
        soil_moisture_param="HS_CV_AVG_-0.2m",
        missing_value=-1000.0
    )

    # Save (all_params = filtered_params + soil moisture)
    output_path = collector.data_dir / "dense_features.npz"
    all_params = filtered_params + ["HS_CV_AVG_-0.2m"]
    save_dense_arrays(
        output_path=str(output_path),
        features_array=features_array,
        mask_array=mask_array,
        station_ids=station_ids,
        date_index=date_index,
        all_params=all_params
    )

    print("\n" + "=" * 70)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nDense array saved to: {output_path}")
    print(f"\nNext: Update SoilMoistureSequenceDataset to use dense arrays")
    print("  - Load dense_features.npz once in __init__")
    print("  - Slice windows for each sequence (instant!)")
    print("  - Expected precomputation time: 2-3 minutes total")
    print("=" * 70)


if __name__ == "__main__":
    main()
