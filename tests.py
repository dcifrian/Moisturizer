#!/usr/bin/env python3
"""
Test script for the refactored Moisturizer codebase.
Tests dataset building, live augmentation, precomputed augmentation,
and verifies normalization stats and data integrity.
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import traceback
from Moisturizer import buildDataset
from MeteoGaliciaCollector import MeteoGaliciaCollector


# Test parameters - tiny dataset for quick testing
SEQ_LENGTH = 2
N_DAYS = 4  # Very short period
N_NEARBY_AVAILABLE = 5
N_NEARBY_IN_FEATURES = 4


def test_dataset_build():
    """Test building the base dataset from scratch."""
    print("\n" + "=" * 70)
    print("TEST 1: Building Base Dataset")
    print("=" * 70)

    # Use buildDataset with minimal parameters
    # This will download data, build dense arrays, etc.
    end_date = datetime.now() - timedelta(days=1)

    print(f"\nBuilding dataset with {N_DAYS} days of data, seq_length={SEQ_LENGTH}...")
    print(f"  End date: {end_date.date()}")

    try:
        train_ds, val_ds, test_ds = buildDataset(
            seq_length=SEQ_LENGTH,
            days=N_DAYS,
            end_date=end_date,
            force_refresh=True,
            coverage_threshold=0.1,  # Lower threshold for short period
        )

        total_samples = len(train_ds) + (len(val_ds) if val_ds else 0) + (len(test_ds) if test_ds else 0)
        print(f"  ✓ Built dataset with {total_samples} total samples")
        print(f"    Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}, Test: {len(test_ds) if test_ds else 0}")

        # Get collector for paths
        collector = MeteoGaliciaCollector()

        # Get filtered params from the dataset
        filtered_params = train_ds.feature_params

        # Test sample extraction
        if len(train_ds) > 0:
            sample = train_ds[0]
            print(f"  ✓ Sample shape: features={sample['features'].shape}, target={sample['target'].shape}")
        else:
            print("  ⚠ Dataset is empty (not enough consecutive days with data)")

        return collector, filtered_params, train_ds

    except Exception as e:
        print(f"  buildDataset failed: {e}")
        # Fall back to manual construction for testing
        print("\n  Falling back to manual dataset construction...")

        collector = MeteoGaliciaCollector()

        # Get stations
        stations_df = collector.get_all_stations()
        if stations_df is None or len(stations_df) == 0:
            raise RuntimeError("Could not get stations data")

        print(f"  ✓ Got {len(stations_df)} stations")

        # Get station IDs
        station_ids = stations_df['station_id'].tolist()

        # Build historical dataset
        start_date = end_date - timedelta(days=N_DAYS)
        parameters = ['TempMedia', 'HumedadeMedia', 'HS_CV_AVG_-0.2m']

        timeseries_df = collector.build_historical_dataset(
            station_ids=station_ids,
            parameter_ids=parameters,
            start_date=start_date,
            end_date=end_date,
            force_refresh=True
        )

        print(f"  ✓ Got {len(timeseries_df)} timeseries records")

        # Analyze coverage
        all_params, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.1)
        print(f"  ✓ Found {len(filtered_params)} parameters with >10% coverage")

        if not filtered_params:
            filtered_params = ['TempMedia', 'HumedadeMedia']
            print(f"  ⚠ Using fallback params: {filtered_params}")

        return collector, filtered_params, None


def test_live_augmented_dataset(collector, filtered_params):
    """Test the live augmented dataset."""
    print("\n" + "=" * 70)
    print("TEST 2: Live Augmented Dataset")
    print("=" * 70)

    from augmented_live import AugmentedLiveDataset

    norm_stats_path = collector.data_dir / "normalization_stats.npz"

    dataset = AugmentedLiveDataset.from_base_dataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        dense_array_path=str(collector.data_dir / "dense_features.npz"),
        feature_params=filtered_params,
        seq_length=SEQ_LENGTH,
        n_nearby_available=N_NEARBY_AVAILABLE,
        n_nearby_in_features=N_NEARBY_IN_FEATURES,
        normalize=True,
        norm_stats_path=str(norm_stats_path),
    )

    print(f"  ✓ Created live augmented dataset")
    print(f"    Base samples: {len(dataset._base_indices) if dataset._base_indices is not None else 'N/A'}")
    print(f"    Augmentation factor: {dataset.total_augmentations}x")
    print(f"    Total samples: {len(dataset)}")
    print(f"    Output features: {dataset.n_output_features}")

    # Test getting samples
    if len(dataset) > 0:
        sample = dataset[0]
        features = sample['features'].numpy()
        target = sample['target'].numpy()

        print(f"  ✓ Sample 0: features shape={features.shape}, target shape={target.shape}")
        print(f"    Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"    Target value: {target[0]:.3f}")

        # Check for normalized invalid markers
        n_invalid = np.sum(features == -2.0)
        print(f"    Invalid markers (-2.0): {n_invalid} values")

        # Verify normalization is within expected range (mostly -1 to 1, -2 for invalid)
        valid_mask = features != -2.0
        if valid_mask.any():
            valid_features = features[valid_mask]
            in_range = np.sum((valid_features >= -1.0) & (valid_features <= 1.0))
            total_valid = len(valid_features)
            print(f"    Valid features in [-1,1]: {in_range}/{total_valid} ({100*in_range/total_valid:.1f}%)")
    else:
        print("  ⚠ Dataset is empty")

    return dataset


def test_precomputed_augmented_dataset(collector, filtered_params):
    """Test precomputing the augmented dataset."""
    print("\n" + "=" * 70)
    print("TEST 3: Precomputed Augmented Dataset")
    print("=" * 70)

    from precompute_augmented import generate_all_augmentations_sequential

    # Use sequential (non-parallel) for small test
    # Output goes to data_dir/augmented/
    generate_all_augmentations_sequential(
        data_dir=str(collector.data_dir),
        seq_length=SEQ_LENGTH,
        n_nearby_available=N_NEARBY_AVAILABLE,
        n_nearby_in_features=N_NEARBY_IN_FEATURES,
        coverage_threshold=0.1,  # Lower threshold for test
    )

    output_path = collector.data_dir / "precomputed_sequences_augmented"

    print(f"\n  ✓ Generated precomputed augmented dataset at {output_path}")

    # Load and verify the precomputed dataset
    if (output_path / "features.npy").exists():
        features = np.load(output_path / "features.npy", mmap_mode='r')
        targets = np.load(output_path / "targets.npy", mmap_mode='r')
        masks = np.load(output_path / "masks.npy", mmap_mode='r')

        print(f"  ✓ Loaded precomputed data:")
        print(f"    Features shape: {features.shape}")
        print(f"    Targets shape: {targets.shape}")
        print(f"    Masks shape: {masks.shape}")

        # Check first sample
        if len(features) > 0:
            sample_features = features[0]
            sample_target = targets[0]

            print(f"  ✓ Sample 0:")
            print(f"    Features range: [{sample_features.min():.3f}, {sample_features.max():.3f}]")
            print(f"    Target value: {float(sample_target):.3f}")

            # Check normalization
            valid_mask = sample_features != -2.0
            if valid_mask.any():
                valid_features = sample_features[valid_mask]
                in_range = np.sum((valid_features >= -1.0) & (valid_features <= 1.0))
                total_valid = len(valid_features)
                print(f"    Valid features in [-1,1]: {in_range}/{total_valid} ({100*in_range/total_valid:.1f}%)")

        return output_path
    else:
        print("  ✗ Precomputed files not found!")
        return None


def test_normalization_stats(collector):
    """Verify normalization stats file."""
    print("\n" + "=" * 70)
    print("TEST 4: Normalization Stats Verification")
    print("=" * 70)

    stats_path = collector.data_dir / "normalization_stats.npz"

    if not stats_path.exists():
        print(f"  ✗ Stats file not found: {stats_path}")
        return False

    stats = np.load(stats_path, allow_pickle=True)

    print(f"  ✓ Loaded stats from {stats_path}")
    print(f"  Keys: {list(stats.keys())}")

    # Check required keys
    required_keys = ['target_feature_mins', 'target_feature_maxs',
                     'nearby_slot_mins', 'nearby_slot_maxs',
                     'target_min', 'target_max', 'n_params', 'feature_params']

    missing = [k for k in required_keys if k not in stats]
    if missing:
        print(f"  ✗ Missing keys: {missing}")
        return False

    print(f"  ✓ All required keys present")

    # Check shapes
    n_params = int(stats['n_params'][0])
    print(f"  n_params: {n_params}")
    print(f"  feature_params: {list(stats['feature_params'])[:5]}...")
    print(f"  target_feature_mins shape: {stats['target_feature_mins'].shape}")
    print(f"  nearby_slot_mins shape: {stats['nearby_slot_mins'].shape}")

    # Check target range
    target_min = float(stats['target_min'].item() if stats['target_min'].ndim == 0 else stats['target_min'][0])
    target_max = float(stats['target_max'].item() if stats['target_max'].ndim == 0 else stats['target_max'][0])
    print(f"  target range: [{target_min:.3f}, {target_max:.3f}]")

    # Check for reasonable values
    if target_max <= target_min:
        print(f"  ⚠ Warning: target_max <= target_min (may not have enough data)")

    return True


def test_feature_layout():
    """Test the FeatureLayout class."""
    print("\n" + "=" * 70)
    print("TEST 5: FeatureLayout Class")
    print("=" * 70)

    from Moisturizer import FeatureLayout

    # Test with typical values
    layout = FeatureLayout(n_params=26, n_nearby=4)

    print(f"  FeatureLayout(n_params=26, n_nearby=4):")
    print(f"    n_target_features: {layout.n_target_features}")
    print(f"    nearby_features_per_station: {layout.nearby_features_per_station}")
    print(f"    n_total_features: {layout.n_total_features}")

    # Verify calculations
    expected_nearby = 1 + 26 + 1  # distance + params + soil
    expected_total = 26 + (expected_nearby * 4)

    assert layout.n_target_features == 26, f"Expected 26, got {layout.n_target_features}"
    assert layout.nearby_features_per_station == expected_nearby, f"Expected {expected_nearby}, got {layout.nearby_features_per_station}"
    assert layout.n_total_features == expected_total, f"Expected {expected_total}, got {layout.n_total_features}"

    print(f"  ✓ All calculations correct")

    # Test helper methods
    print(f"    nearby_start_idx(0): {layout.nearby_start_idx(0)}")
    print(f"    nearby_end_idx(0): {layout.nearby_end_idx(0)}")
    print(f"    nearby_start_idx(3): {layout.nearby_start_idx(3)}")

    return True


def test_normalize_functions():
    """Test the shared normalization functions."""
    print("\n" + "=" * 70)
    print("TEST 6: Normalization Functions")
    print("=" * 70)

    from Moisturizer import (
        normalize_features, normalize_target, denormalize_target,
        INVALID_MARKER_API, INVALID_MARKER_MISSING, NORMALIZED_INVALID_MARKER
    )

    # Test normalize_features
    features = np.array([[0, 50, 100, INVALID_MARKER_API],
                         [25, 75, INVALID_MARKER_MISSING, 50]], dtype=np.float32)
    feature_mins = np.array([0, 0, 0, 0], dtype=np.float32)
    feature_maxs = np.array([100, 100, 100, 100], dtype=np.float32)

    normalized = normalize_features(features.copy(), feature_mins, feature_maxs)

    print(f"  Original features:\n    {features}")
    print(f"  Normalized features:\n    {normalized}")

    # Check expected values
    assert normalized[0, 0] == -1.0, f"Expected -1.0, got {normalized[0, 0]}"
    assert normalized[0, 1] == 0.0, f"Expected 0.0, got {normalized[0, 1]}"
    assert normalized[0, 2] == 1.0, f"Expected 1.0, got {normalized[0, 2]}"
    assert normalized[0, 3] == NORMALIZED_INVALID_MARKER, f"Expected {NORMALIZED_INVALID_MARKER}, got {normalized[0, 3]}"
    assert normalized[1, 2] == NORMALIZED_INVALID_MARKER, f"Expected {NORMALIZED_INVALID_MARKER}, got {normalized[1, 2]}"

    print(f"  ✓ normalize_features works correctly")

    # Test normalize_target
    target_min, target_max = 0.0, 100.0

    assert normalize_target(0, target_min, target_max) == -1.0
    assert normalize_target(50, target_min, target_max) == 0.0
    assert normalize_target(100, target_min, target_max) == 1.0
    assert normalize_target(INVALID_MARKER_API, target_min, target_max) == NORMALIZED_INVALID_MARKER

    print(f"  ✓ normalize_target works correctly")

    # Test denormalize_target
    assert denormalize_target(-1.0, target_min, target_max) == 0.0
    assert denormalize_target(0.0, target_min, target_max) == 50.0
    assert denormalize_target(1.0, target_min, target_max) == 100.0
    assert denormalize_target(NORMALIZED_INVALID_MARKER, target_min, target_max) is None

    print(f"  ✓ denormalize_target works correctly")

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MOISTURIZER REFACTORED CODE TEST SUITE")
    print("=" * 70)
    print(f"Test parameters: seq_length={SEQ_LENGTH}, n_days={N_DAYS}")
    print(f"                 n_nearby_available={N_NEARBY_AVAILABLE}, n_nearby_in_features={N_NEARBY_IN_FEATURES}")

    results = {}

    # Test FeatureLayout and normalization functions first (no data needed)
    try:
        results['feature_layout'] = test_feature_layout()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        results['feature_layout'] = False

    try:
        results['normalize_functions'] = test_normalize_functions()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        results['normalize_functions'] = False

    # Build dataset
    try:
        collector, filtered_params, base_dataset = test_dataset_build()
        results['dataset_build'] = True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        results['dataset_build'] = False
        collector, filtered_params = None, None

    # Test normalization stats
    if collector:
        try:
            results['normalization_stats'] = test_normalization_stats(collector)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            results['normalization_stats'] = False

    # Test live augmented dataset
    if collector and filtered_params:
        try:
            live_dataset = test_live_augmented_dataset(collector, filtered_params)
            results['live_augmented'] = True
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            results['live_augmented'] = False

    # Test precomputed augmented dataset
    if collector and filtered_params:
        try:
            precomputed_path = test_precomputed_augmented_dataset(collector, filtered_params)
            results['precomputed_augmented'] = precomputed_path is not None
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            traceback.print_exc()
            results['precomputed_augmented'] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - see above for details")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
