#!/usr/bin/env python3
"""
Debug the specific issue where the example sample has target=-1000

This script investigates:
1. What is sample_index[0]?
2. What does the dense array contain for that station/date?
3. Why is the target coming back as -1000?
"""

import numpy as np
import pandas as pd
from pathlib import Path

def main():
    print("=" * 70)
    print("DEBUGGING EXAMPLE SAMPLE WITH TARGET=-1000")
    print("=" * 70)

    # Load dense array
    dense_path = Path("meteogalicia_data/dense_features.npz")
    if not dense_path.exists():
        print(f"✗ File not found: {dense_path}")
        print("Please run this in the directory with your full dataset.")
        return

    data = np.load(dense_path)
    features = data['features']
    masks = data['masks']
    station_ids = data['station_ids']
    dates = pd.to_datetime(data['dates'])
    feature_params = data['feature_params']

    print(f"\nDense array info:")
    print(f"  Shape: {features.shape}")
    print(f"  Dates: {len(dates)} from {dates[0].date()} to {dates[-1].date()}")
    print(f"  Stations: {len(station_ids)}")

    # Find soil moisture index
    soil_idx = len(feature_params) - 1  # Should be last
    print(f"  Soil moisture param: {feature_params[soil_idx]} at index {soil_idx}")

    # Get first station (19005 based on user's output)
    first_station_id = 19005
    if first_station_id not in station_ids:
        first_station_id = station_ids[0]
        print(f"\n  Note: Station 19005 not found, using {first_station_id}")

    station_idx = np.where(station_ids == first_station_id)[0][0]
    print(f"\nAnalyzing station {first_station_id} (index {station_idx}):")

    # Get all soil moisture values for this station
    soil_values = features[station_idx, :, soil_idx]
    soil_mask = masks[station_idx, :, soil_idx]

    print(f"\n  Soil moisture values for station {first_station_id}:")
    print(f"  {'Date':<12} {'Value':<10} {'Masked':<8} {'Valid?'}")
    print(f"  " + "-" * 50)

    for i, (date, value, mask) in enumerate(zip(dates, soil_values, soil_mask)):
        masked = "Yes" if mask == 1.0 else "No"
        valid = "✓" if (0.0 <= value <= 1.0 and mask == 1.0) else "✗"
        if value == -1000.0:
            value_str = "-1000.0 (OUR MARKER)"
        elif value == -9999.0:
            value_str = "-9999.0 (METEOGAL)"
        else:
            value_str = f"{value:.3f}"
        print(f"  {date.date()} {value_str:<20} {masked:<8} {valid}")

    # Count statistics
    valid_mask = (soil_mask == 1.0)
    valid_values = soil_values[valid_mask]
    in_range = ((valid_values >= 0.0) & (valid_values <= 1.0))

    print(f"\n  Statistics:")
    print(f"    Total dates: {len(dates)}")
    print(f"    Masked as valid: {valid_mask.sum()}")
    print(f"    In range [0,1]: {in_range.sum()}")
    print(f"    Invalid/missing: {(~in_range).sum()}")

    if len(valid_values) > 0:
        print(f"    Value range: [{valid_values.min():.3f}, {valid_values.max():.3f}]")

    # Check what samples would be built
    print(f"\n" + "=" * 70)
    print("CHECKING SAMPLE BUILDING")
    print("=" * 70)

    # Simulate sample_index building for this station
    # Based on _build_sample_index logic
    seq_length = 64
    samples_for_station = []

    for date_idx in range(len(dates)):
        end_date = dates[date_idx]
        start_idx = date_idx - seq_length + 1

        if start_idx < 0:
            continue

        # Check if target is valid
        target_value = soil_values[date_idx]
        target_mask = soil_mask[date_idx]

        # Check for invalid markers
        invalid_markers = {-9999.0, -1000.0, 9999.0}
        is_invalid = any(abs(target_value - marker) < 0.01 for marker in invalid_markers)

        if target_mask != 1.0:
            reason = "not masked"
        elif is_invalid:
            reason = f"invalid marker ({target_value})"
        elif target_value < 0.0 or target_value > 1.0:
            reason = f"out of range ({target_value})"
        else:
            reason = "VALID ✓"
            samples_for_station.append({
                'end_date': end_date,
                'target': target_value
            })

        if date_idx < 70:  # Show first samples
            print(f"  End date {end_date.date()}: target={target_value:.3f}, "
                  f"masked={target_mask==1.0}, reason={reason}")

    print(f"\n  Total valid samples for station {first_station_id}: {len(samples_for_station)}")

    if samples_for_station:
        print(f"\n  First 5 valid samples:")
        for i, sample in enumerate(samples_for_station[:5]):
            print(f"    Sample {i}: end_date={sample['end_date'].date()}, "
                  f"target={sample['target']:.3f}")

        print(f"\n  First sample target: {samples_for_station[0]['target']:.3f}")
        if samples_for_station[0]['target'] == -1000.0:
            print(f"    ✗ ERROR: First sample has -1000 target!")
            print(f"    This should have been filtered in _build_sample_index()")
        elif samples_for_station[0]['target'] < 0.0 or samples_for_station[0]['target'] > 1.0:
            print(f"    ✗ ERROR: First sample target outside [0, 1]!")
        else:
            print(f"    ✓ First sample target is valid")


if __name__ == "__main__":
    main()
