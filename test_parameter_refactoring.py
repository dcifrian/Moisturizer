#!/usr/bin/env python3
"""
Test the refactored parameter coverage and validation functionality
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from Moisturizer import MeteoGaliciaCollector


def test_analyze_parameter_coverage():
    """Test the refactored analyze_parameter_coverage() method"""
    print("=" * 60)
    print("TEST 1: analyze_parameter_coverage() with timeseries_df")
    print("=" * 60)

    collector = MeteoGaliciaCollector()

    # Check if required files exist
    if not collector.timeseries_file.exists():
        print(f"✗ Timeseries file not found: {collector.timeseries_file}")
        print("  Skipping this test")
        return False

    if not collector.stations_file.exists():
        print(f"✗ Stations file not found: {collector.stations_file}")
        print("  Skipping this test")
        return False

    # Load data
    print(f"Loading timeseries from {collector.timeseries_file}...")
    timeseries_df = pd.read_csv(collector.timeseries_file)
    timeseries_df['date'] = pd.to_datetime(timeseries_df['date'])

    print(f"Loading stations from {collector.stations_file}...")
    stations_df = pd.read_csv(collector.stations_file)

    # Test Method 1: Analyze from timeseries_df
    try:
        coverage_dict, filtered_params = collector.analyze_parameter_coverage(
            timeseries_df=timeseries_df,
            stations_df=stations_df,
            coverage_threshold=0.25,
            soil_moisture_param="HS_CV_AVG_-0.2m",
            add_coordinate_features=True
        )

        print(f"\n✓ Method 1 (timeseries_df) succeeded!")
        print(f"  Found {len(filtered_params)} filtered parameters")
        print(f"  Parameters: {filtered_params[:5]}..." if len(filtered_params) > 5 else f"  Parameters: {filtered_params}")

        # Verify coordinate features were added
        coordinate_features = ['altitude', 'utmx', 'utmy']
        has_coords = all(coord in filtered_params for coord in coordinate_features)
        if has_coords:
            print(f"  ✓ Coordinate features included: {coordinate_features}")
        else:
            print(f"  ✗ Missing coordinate features!")
            return False

        return True

    except Exception as e:
        print(f"\n✗ Method 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_params_validation():
    """Test that feature_params validation works when loading precomputed data"""
    print("\n" + "=" * 60)
    print("TEST 2: Feature params validation")
    print("=" * 60)

    collector = MeteoGaliciaCollector()

    # Check if precomputed data exists
    precomputed_path = collector.data_dir / "precomputed_sequences"
    if not precomputed_path.exists():
        print(f"✗ Precomputed data not found: {precomputed_path}")
        print("  Skipping this test - run buildDataset first to create precomputed data")
        return None

    # Check if feature_params.npy exists
    feature_params_file = precomputed_path / "feature_params.npy"
    if not feature_params_file.exists():
        print(f"✗ feature_params.npy not found in {precomputed_path}")
        print("  This is expected if you haven't rebuilt the precomputed data yet")
        print("  The validation will work once you rebuild the precomputed data")
        return None

    # Load the saved feature_params
    saved_params = np.load(feature_params_file).tolist()
    print(f"Precomputed dataset has {len(saved_params)} feature parameters")
    print(f"First 5 params: {saved_params[:5]}")

    # Test 1: Load with None (should succeed and use saved params)
    print("\nTest 2a: Loading with feature_params=None")
    try:
        from Moisturizer import SoilMoistureSequenceDataset

        dataset = SoilMoistureSequenceDataset(
            timeseries=str(collector.timeseries_file),
            stations=str(collector.stations_file),
            nearest=str(collector.nearest_file),
            seq_length=64,
            n_nearest=4,
            feature_params=None,  # Should use precomputed params
            precomputed_path=str(precomputed_path)
        )

        print(f"✓ Successfully loaded with feature_params=None")
        print(f"  Dataset is using {len(dataset.feature_params)} feature parameters")

        if dataset.feature_params == saved_params:
            print(f"  ✓ Feature params match saved params")
        else:
            print(f"  ✗ Feature params don't match!")
            return False

    except Exception as e:
        print(f"✗ Failed to load with feature_params=None: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 2: Load with matching params (should succeed)
    print("\nTest 2b: Loading with matching feature_params")
    try:
        dataset = SoilMoistureSequenceDataset(
            timeseries=str(collector.timeseries_file),
            stations=str(collector.stations_file),
            nearest=str(collector.nearest_file),
            seq_length=64,
            n_nearest=4,
            feature_params=saved_params,  # Should match
            precomputed_path=str(precomputed_path)
        )

        print(f"✓ Successfully loaded with matching feature_params")

    except Exception as e:
        print(f"✗ Failed to load with matching params: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 3: Load with different params (should fail)
    print("\nTest 2c: Loading with different feature_params (should fail)")
    try:
        different_params = saved_params[:10]  # Use only first 10 params

        dataset = SoilMoistureSequenceDataset(
            timeseries=str(collector.timeseries_file),
            stations=str(collector.stations_file),
            nearest=str(collector.nearest_file),
            seq_length=64,
            n_nearest=4,
            feature_params=different_params,  # Should NOT match
            precomputed_path=str(precomputed_path)
        )

        print(f"✗ Should have failed but didn't!")
        return False

    except ValueError as e:
        if "Feature parameters mismatch" in str(e):
            print(f"✓ Correctly rejected mismatched feature_params")
            print(f"  Error message: {str(e).split(chr(10))[0]}...")
        else:
            print(f"✗ Wrong error: {e}")
            return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == '__main__':
    print("Testing parameter coverage refactoring and validation\n")

    # Test 1: analyze_parameter_coverage
    test1_result = test_analyze_parameter_coverage()

    # Test 2: feature_params validation
    test2_result = test_feature_params_validation()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Test 1 (analyze_parameter_coverage): {'✓ PASS' if test1_result else '✗ FAIL'}")

    if test2_result is None:
        print(f"Test 2 (feature_params validation): ⊘ SKIPPED (no precomputed data)")
    else:
        print(f"Test 2 (feature_params validation): {'✓ PASS' if test2_result else '✗ FAIL'}")

    # Exit code
    if test1_result and (test2_result is None or test2_result):
        print("\n✓ All tests passed (or skipped)")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
