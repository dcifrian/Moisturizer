"""
Test script for parameter coverage analysis
Demonstrates:
1. analyze_parameter_coverage() function
2. Using filtered parameters with SoilMoistureSequenceDataset
3. Fixed timestamp conversion in __getitem__
"""
import sys
sys.path.insert(0, '/home/user/Moisturizer')

from Moisturizer import MeteoGaliciaCollector

def test_parameter_coverage():
    """Test the parameter coverage analysis function"""
    print("=" * 70)
    print("TEST: Parameter Coverage Analysis")
    print("=" * 70)

    collector = MeteoGaliciaCollector()

    # Check if ML dataset exists
    ml_dataset_file = collector.cache_dir / "ml_ready_dataset.csv"

    if not ml_dataset_file.exists():
        print(f"\n✗ ML dataset not found at {ml_dataset_file}")
        print("  Please run the main script first to generate the dataset.")
        print("  Run: python3 Moisturizer.py")
        return None, None

    # Analyze coverage with 25% threshold
    coverage, filtered_params = collector.analyze_parameter_coverage(
        coverage_threshold=0.25
    )

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Total unique parameters: {len(coverage)}")
    print(f"  Parameters with ≥25% coverage: {len(filtered_params)}")
    print(f"  Filtered out: {len(coverage) - len(filtered_params)}")

    # Show a few examples of filtered vs kept
    kept = [p for p in coverage.keys() if p in filtered_params]
    filtered_out = [p for p in coverage.keys() if p not in filtered_params]

    if kept:
        print(f"\n  Examples of KEPT parameters:")
        for param in sorted(kept)[:5]:
            print(f"    - {param}: {coverage[param]*100:.1f}%")

    if filtered_out:
        print(f"\n  Examples of FILTERED OUT parameters:")
        for param in sorted(filtered_out)[:5]:
            print(f"    - {param}: {coverage[param]*100:.1f}%")

    return coverage, filtered_params


def test_dataset_with_filtered_params():
    """Test using filtered parameters with SoilMoistureSequenceDataset"""
    print("\n" + "=" * 70)
    print("TEST: Dataset with Filtered Parameters")
    print("=" * 70)

    collector = MeteoGaliciaCollector()

    # Get filtered parameters
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)

    if not filtered_params:
        print("\n✗ No parameters passed the threshold!")
        return

    print(f"\nUsing {len(filtered_params)} filtered parameters...")

    try:
        # Import will fail if PyTorch not available, but that's expected
        from Moisturizer import SoilMoistureSequenceDataset

        # Try creating dataset with filtered params
        dataset = SoilMoistureSequenceDataset(
            timeseries=str(collector.timeseries_file),
            stations=str(collector.stations_file),
            nearest=str(collector.nearest_file),
            seq_length=7,
            n_nearest=4,
            feature_params=filtered_params  # Use filtered parameters!
        )

        print(f"\n✓ Dataset created successfully!")
        print(f"  Sequence length: {dataset.seq_length}")
        print(f"  Target stations: {len(dataset.target_stations)}")
        print(f"  Feature parameters: {len(dataset.feature_params)}")
        print(f"  Valid samples: {len(dataset.sample_index)}")

        # Try getting a sample (tests timestamp conversion fix)
        if len(dataset) > 0:
            print(f"\nTesting __getitem__ (timestamp conversion fix)...")
            sample = dataset[0]
            print(f"✓ Sample retrieved successfully!")
            print(f"  Features shape: {sample['features'].shape}")
            print(f"  Target: {sample['target']}")
            print(f"  Station ID: {sample['target_station_id']}")
            print(f"  End date (Unix timestamp): {sample['end_date']}")

            # Verify end_date is a float, not pandas Timestamp
            if isinstance(sample['end_date'], float):
                print(f"✓ Timestamp correctly converted to float!")
            else:
                print(f"✗ WARNING: end_date is {type(sample['end_date'])}, expected float")

    except ImportError:
        print("\n⚠ PyTorch not available - cannot test Dataset class")
        print("  But parameter filtering would work when PyTorch is installed!")
    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    coverage, filtered_params = test_parameter_coverage()

    if filtered_params:
        test_dataset_with_filtered_params()

    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)
