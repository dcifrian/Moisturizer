#!/usr/bin/env python3
"""
Check validity of soil moisture targets in dense arrays and precomputed sequences.

Reports:
- How many targets are outside [0.0, 1.0] range
- Per-station breakdown
- Examples of invalid targets with station/date info
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def check_dense_array():
    """Check soil moisture values in dense_features.npz"""
    print("=" * 70)
    print("CHECKING DENSE ARRAY (dense_features.npz)")
    print("=" * 70)

    dense_path = Path("meteogalicia_data/dense_features.npz")
    if not dense_path.exists():
        print(f"✗ File not found: {dense_path}")
        return

    # Load dense array
    data = np.load(dense_path)
    features = data['features']  # Shape: (stations, dates, features)
    masks = data['masks']
    station_ids = data['station_ids']
    dates = pd.to_datetime(data['dates'])
    feature_params = data['feature_params']

    print(f"\nDense array shape: {features.shape}")
    print(f"  Stations: {len(station_ids)}")
    print(f"  Dates: {len(dates)}")
    print(f"  Features: {len(feature_params)}")

    # Find soil moisture index (should be last feature)
    soil_idx = None
    for i, param in enumerate(feature_params):
        if 'HS_CV_AVG' in param or 'soil' in param.lower():
            soil_idx = i
            print(f"\nSoil moisture parameter: {param} (index {i})")
            break

    if soil_idx is None:
        print("✗ Could not find soil moisture parameter!")
        return

    # Extract soil moisture values for all stations
    soil_moisture = features[:, :, soil_idx]  # Shape: (stations, dates)
    soil_mask = masks[:, :, soil_idx]  # Shape: (stations, dates)

    print(f"\n" + "=" * 70)
    print("ANALYZING SOIL MOISTURE VALUES")
    print("=" * 70)

    # Per-station statistics
    station_stats = []
    invalid_examples = []

    for station_idx, station_id in enumerate(station_ids):
        # Get all soil moisture values for this station
        station_values = soil_moisture[station_idx, :]
        station_mask = soil_mask[station_idx, :]

        # Count valid (masked) values
        valid_mask = station_mask == 1.0
        valid_values = station_values[valid_mask]

        if len(valid_values) == 0:
            continue

        # Check for values outside [0, 1]
        outside_range = (valid_values < 0.0) | (valid_values > 1.0)
        is_missing_marker = np.abs(valid_values + 1000.0) < 0.01  # Check for -1000
        is_invalid_marker = np.abs(valid_values + 9999.0) < 0.01  # Check for -9999

        num_outside = outside_range.sum()
        num_missing = is_missing_marker.sum()
        num_invalid = is_invalid_marker.sum()

        # Record statistics
        station_stats.append({
            'station_id': int(station_id),
            'total_valid': len(valid_values),
            'outside_range': int(num_outside),
            'has_minus_1000': int(num_missing),
            'has_minus_9999': int(num_invalid),
            'min_value': float(valid_values.min()),
            'max_value': float(valid_values.max())
        })

        # Collect examples of invalid values
        if num_outside > 0:
            # Find indices of invalid values
            date_indices = np.where(valid_mask)[0]
            invalid_indices = date_indices[outside_range]

            # Take up to 3 examples per station
            for idx in invalid_indices[:3]:
                invalid_examples.append({
                    'station_id': int(station_id),
                    'date': dates[idx],
                    'value': float(station_values[idx]),
                    'is_minus_1000': abs(station_values[idx] + 1000.0) < 0.01,
                    'is_minus_9999': abs(station_values[idx] + 9999.0) < 0.01
                })

    # Print summary
    df_stats = pd.DataFrame(station_stats)

    print(f"\nPER-STATION SUMMARY:")
    print(f"  Total stations with soil moisture: {len(df_stats)}")

    stations_with_invalid = df_stats[df_stats['outside_range'] > 0]
    print(f"\n  Stations with values outside [0, 1]: {len(stations_with_invalid)}")

    if len(stations_with_invalid) > 0:
        print(f"\n  Details:")
        for _, row in stations_with_invalid.iterrows():
            print(f"    Station {row['station_id']}: "
                  f"{row['outside_range']}/{row['total_valid']} invalid "
                  f"(min={row['min_value']:.3f}, max={row['max_value']:.3f})")
            if row['has_minus_1000'] > 0:
                print(f"      → Contains {row['has_minus_1000']} values of -1000.0")
            if row['has_minus_9999'] > 0:
                print(f"      → Contains {row['has_minus_9999']} values of -9999.0")

    # Print examples
    if invalid_examples:
        print(f"\n  Example invalid values:")
        for ex in invalid_examples[:10]:
            marker = ""
            if ex['is_minus_1000']:
                marker = " (MISSING_VALUE=-1000)"
            elif ex['is_minus_9999']:
                marker = " (INVALID_MARKER=-9999)"
            print(f"    Station {ex['station_id']}, {ex['date'].date()}: "
                  f"{ex['value']:.3f}{marker}")

    # Overall statistics
    total_valid = df_stats['total_valid'].sum()
    total_outside = df_stats['outside_range'].sum()
    total_minus_1000 = df_stats['has_minus_1000'].sum()
    total_minus_9999 = df_stats['has_minus_9999'].sum()

    print(f"\n" + "=" * 70)
    print("OVERALL STATISTICS (DENSE ARRAY)")
    print("=" * 70)
    print(f"  Total valid soil moisture values: {total_valid:,}")
    print(f"  Values outside [0, 1]: {total_outside:,} ({100*total_outside/total_valid:.2f}%)")
    print(f"    - Values = -1000.0: {total_minus_1000:,}")
    print(f"    - Values = -9999.0: {total_minus_9999:,}")
    print(f"  Values in [0, 1]: {total_valid - total_outside:,} ({100*(total_valid-total_outside)/total_valid:.2f}%)")


def check_precomputed_sequences():
    """Check targets in precomputed_sequences.npz"""
    print("\n\n" + "=" * 70)
    print("CHECKING PRECOMPUTED SEQUENCES (precomputed_sequences.npz)")
    print("=" * 70)

    precomp_path = Path("meteogalicia_data/precomputed_sequences.npz")
    if not precomp_path.exists():
        print(f"✗ File not found: {precomp_path}")
        return

    # Load precomputed data
    data = np.load(precomp_path)
    targets = data['targets']  # Shape: (num_samples, 1)
    target_stations = data['target_stations']  # Shape: (num_samples,)
    end_dates = pd.to_datetime(data['end_dates'], unit='s')

    print(f"\nPrecomputed sequences shape: {targets.shape}")
    print(f"  Total samples: {len(targets)}")

    # Flatten targets for easier analysis
    targets_flat = targets.flatten()

    # Check if data is normalized
    is_normalized = data.get('is_normalized', np.array([False]))[0]
    print(f"  Data is normalized: {is_normalized}")

    if is_normalized:
        print(f"\n  NOTE: Data is normalized to [-1, 1] range")
        print(f"  Checking for invalid markers in normalized space...")

        # In normalized space, -2.0 is the invalid marker
        invalid_marker = -2.0
        is_invalid = np.abs(targets_flat - invalid_marker) < 0.01

        print(f"\n  Target statistics (normalized):")
        print(f"    Min: {targets_flat.min():.3f}")
        print(f"    Max: {targets_flat.max():.3f}")
        print(f"    Invalid markers (-2.0): {is_invalid.sum():,} ({100*is_invalid.sum()/len(targets_flat):.2f}%)")

        # Per-station breakdown
        station_stats = defaultdict(lambda: {'total': 0, 'invalid': 0, 'min': float('inf'), 'max': float('-inf')})

        for i, (target, station_id) in enumerate(zip(targets_flat, target_stations)):
            station_stats[station_id]['total'] += 1
            if abs(target - invalid_marker) < 0.01:
                station_stats[station_id]['invalid'] += 1
            else:
                station_stats[station_id]['min'] = min(station_stats[station_id]['min'], target)
                station_stats[station_id]['max'] = max(station_stats[station_id]['max'], target)

        print(f"\n  Per-station breakdown:")
        stations_with_invalid = []
        for station_id in sorted(station_stats.keys()):
            stats = station_stats[station_id]
            if stats['invalid'] > 0:
                stations_with_invalid.append(station_id)
                print(f"    Station {station_id}: {stats['invalid']}/{stats['total']} invalid "
                      f"({100*stats['invalid']/stats['total']:.1f}%)")

        if not stations_with_invalid:
            print(f"    ✓ No stations with invalid targets!")

    else:
        print(f"\n  NOTE: Data is NOT normalized (raw values)")
        print(f"  Expected range: [0.0, 1.0]")

        # Check for values outside [0, 1]
        outside_range = (targets_flat < 0.0) | (targets_flat > 1.0)
        is_minus_1000 = np.abs(targets_flat + 1000.0) < 0.01
        is_minus_9999 = np.abs(targets_flat + 9999.0) < 0.01

        print(f"\n  Target statistics (raw):")
        print(f"    Min: {targets_flat.min():.3f}")
        print(f"    Max: {targets_flat.max():.3f}")
        print(f"    Outside [0, 1]: {outside_range.sum():,} ({100*outside_range.sum()/len(targets_flat):.2f}%)")
        print(f"    Values = -1000.0: {is_minus_1000.sum():,}")
        print(f"    Values = -9999.0: {is_minus_9999.sum():,}")

        # Per-station breakdown
        station_stats = defaultdict(lambda: {'total': 0, 'outside': 0, 'minus_1000': 0, 'minus_9999': 0})
        invalid_examples = []

        for i, (target, station_id) in enumerate(zip(targets_flat, target_stations)):
            station_stats[station_id]['total'] += 1
            if target < 0.0 or target > 1.0:
                station_stats[station_id]['outside'] += 1
                if abs(target + 1000.0) < 0.01:
                    station_stats[station_id]['minus_1000'] += 1
                if abs(target + 9999.0) < 0.01:
                    station_stats[station_id]['minus_9999'] += 1

                # Collect examples
                if len(invalid_examples) < 10:
                    invalid_examples.append({
                        'station_id': int(station_id),
                        'date': end_dates[i],
                        'value': float(target),
                        'is_minus_1000': abs(target + 1000.0) < 0.01,
                        'is_minus_9999': abs(target + 9999.0) < 0.01
                    })

        print(f"\n  Per-station breakdown:")
        stations_with_invalid = []
        for station_id in sorted(station_stats.keys()):
            stats = station_stats[station_id]
            if stats['outside'] > 0:
                stations_with_invalid.append(station_id)
                print(f"    Station {station_id}: {stats['outside']}/{stats['total']} invalid "
                      f"({100*stats['outside']/stats['total']:.1f}%)")
                if stats['minus_1000'] > 0:
                    print(f"      → {stats['minus_1000']} values = -1000.0")
                if stats['minus_9999'] > 0:
                    print(f"      → {stats['minus_9999']} values = -9999.0")

        if not stations_with_invalid:
            print(f"    ✓ No stations with invalid targets!")

        # Print examples
        if invalid_examples:
            print(f"\n  Example invalid values:")
            for ex in invalid_examples:
                marker = ""
                if ex['is_minus_1000']:
                    marker = " (MISSING_VALUE=-1000)"
                elif ex['is_minus_9999']:
                    marker = " (INVALID_MARKER=-9999)"
                print(f"    Station {ex['station_id']}, {ex['date'].date()}: "
                      f"{ex['value']:.3f}{marker}")


if __name__ == "__main__":
    print("TARGET VALIDITY CHECK")
    print("=" * 70)
    print("Checking for invalid soil moisture targets in:")
    print("  1. Dense feature array (dense_features.npz)")
    print("  2. Precomputed sequences (precomputed_sequences.npz)")
    print()

    check_dense_array()
    check_precomputed_sequences()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
