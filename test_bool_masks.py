#!/usr/bin/env python3
"""
Test that masks are now using bool dtype instead of float32
This should save 75% memory on masks
"""

import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from Moisturizer import MeteoGaliciaCollector, build_dense_feature_array
import pandas as pd


def test_dense_array_masks():
    """Test that build_dense_feature_array produces boolean masks"""
    print("=" * 60)
    print("TEST 1: build_dense_feature_array mask dtype")
    print("=" * 60)

    collector = MeteoGaliciaCollector()

    # Check if required files exist
    if not collector.timeseries_file.exists():
        print(f"✗ Skipping: timeseries file not found")
        return None

    if not collector.stations_file.exists():
        print(f"✗ Skipping: stations file not found")
        return None

    # Load data
    print("Loading data...")
    timeseries_df = pd.read_csv(collector.timeseries_file)
    timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])
    stations_df = pd.read_csv(collector.stations_file)

    # Get filtered params
    _, filtered_params = collector.analyze_parameter_coverage(
        timeseries_df=timeseries_df,
        stations_df=stations_df,
        coverage_threshold=0.25
    )

    print(f"\nBuilding dense array with {len(filtered_params)} parameters...")

    # Build dense array
    features_array, mask_array, station_ids, date_index = build_dense_feature_array(
        timeseries_df=timeseries_df,
        stations_df=stations_df,
        feature_params=filtered_params,
        soil_moisture_param="HS_CV_AVG_-0.2m"
    )

    # Check mask dtype
    print(f"\n✓ Features array dtype: {features_array.dtype} (expected: float32)")
    print(f"✓ Mask array dtype: {mask_array.dtype} (expected: bool)")

    if mask_array.dtype == bool:
        print(f"✓ PASS: Mask is boolean!")
    else:
        print(f"✗ FAIL: Mask is {mask_array.dtype}, not bool!")
        return False

    # Check memory savings
    old_size = mask_array.size * 4  # float32 would be 4 bytes
    new_size = mask_array.nbytes    # bool is 1 byte
    savings = (old_size - new_size) / old_size * 100

    print(f"\n✓ Memory usage:")
    print(f"  Old (float32): {old_size / 1e6:.1f} MB")
    print(f"  New (bool):    {new_size / 1e6:.1f} MB")
    print(f"  Savings:       {savings:.1f}% ({(old_size - new_size) / 1e6:.1f} MB)")

    # Check that mask values are True/False (not 0.0/1.0)
    unique_values = np.unique(mask_array)
    print(f"\n✓ Unique mask values: {unique_values} (expected: [False, True])")

    if set(unique_values) == {False, True}:
        print(f"✓ PASS: Mask contains only boolean values!")
    else:
        print(f"✗ FAIL: Mask contains unexpected values: {unique_values}")
        return False

    # Check that mask operations still work
    valid_count = mask_array.sum()
    coverage = valid_count / mask_array.size * 100
    print(f"\n✓ Data coverage: {coverage:.1f}% ({valid_count:,} / {mask_array.size:,})")

    return True


def test_precomputed_masks():
    """Test that precomputed data has boolean masks"""
    print("\n" + "=" * 60)
    print("TEST 2: Precomputed data mask dtype")
    print("=" * 60)

    collector = MeteoGaliciaCollector()
    precomputed_path = collector.data_dir / "precomputed_sequences"

    if not precomputed_path.exists():
        print("✗ Skipping: No precomputed data found")
        print("  Run buildDataset to create precomputed data with new bool masks")
        return None

    masks_file = precomputed_path / "masks.npy"
    if not masks_file.exists():
        print("✗ Skipping: No masks.npy found")
        return None

    print("Loading precomputed masks...")
    masks = np.load(masks_file, mmap_mode='r')

    print(f"✓ Mask dtype: {masks.dtype}")

    if masks.dtype == bool:
        print(f"✓ PASS: Precomputed masks are boolean!")
    elif masks.dtype == np.float32:
        print(f"⚠ WARNING: Precomputed masks are still float32")
        print(f"  This is expected if you haven't rebuilt the dataset yet")
        print(f"  Rebuild with buildDataset to get the memory savings")
        return None
    else:
        print(f"✗ FAIL: Unexpected dtype: {masks.dtype}")
        return False

    # Memory savings
    old_size = masks.size * 4
    new_size = masks.nbytes
    savings = (old_size - new_size) / old_size * 100

    print(f"\n✓ Memory savings on precomputed masks:")
    print(f"  Old (float32): {old_size / 1e6:.1f} MB")
    print(f"  New (bool):    {new_size / 1e6:.1f} MB")
    print(f"  Savings:       {savings:.1f}% ({(old_size - new_size) / 1e6:.1f} MB)")

    return True


if __name__ == '__main__':
    print("Testing boolean mask implementation\n")

    test1_result = test_dense_array_masks()
    test2_result = test_precomputed_masks()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if test1_result:
        print("✓ TEST 1 (build_dense_feature_array): PASS")
    elif test1_result is None:
        print("⊘ TEST 1 (build_dense_feature_array): SKIPPED")
    else:
        print("✗ TEST 1 (build_dense_feature_array): FAIL")

    if test2_result:
        print("✓ TEST 2 (precomputed data): PASS")
    elif test2_result is None:
        print("⊘ TEST 2 (precomputed data): SKIPPED (rebuild dataset to test)")
    else:
        print("✗ TEST 2 (precomputed data): FAIL")

    if test1_result and (test2_result or test2_result is None):
        print("\n✓ All tests passed! Masks are now boolean (75% memory savings)")
        sys.exit(0)
    elif test1_result is None:
        print("\n⊘ Tests skipped (missing data)")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
