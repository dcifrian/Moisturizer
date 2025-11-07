#!/usr/bin/env python3
"""
Regenerate the entire dataset from scratch with proper normalization

This script:
1. Analyzes parameter coverage
2. Generates sample index for all valid sequences
3. Pre-computes and normalizes all sequences
4. Saves everything with correct normalization stats

Run this once, wait ~24 hours, then enjoy 10,000+ samples/sec training!
"""

import argparse
from pathlib import Path
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

def regenerate_dataset(
    coverage_threshold=0.25,
    seq_length=64,
    n_nearest=4,
    data_dir="./meteogalicia_data"
):
    """
    Regenerate dataset from scratch with proper normalization

    Args:
        coverage_threshold: Minimum parameter coverage (default 0.25 = 25%)
        seq_length: Sequence length in days (default 64)
        n_nearest: Number of nearest stations to include (default 4)
        data_dir: Data directory path
    """
    print("=" * 70)
    print("REGENERATING DATASET FROM SCRATCH")
    print("=" * 70)
    print(f"This will take ~24 hours but ensures clean, correct data!")
    print("=" * 70)

    # Initialize collector
    print("\n1. Initializing data collector...")
    collector = MeteoGaliciaCollector(data_dir=data_dir)

    # Analyze parameter coverage
    print("\n2. Analyzing parameter coverage...")
    coverage, filtered_params = collector.analyze_parameter_coverage(
        coverage_threshold=coverage_threshold,
        soil_moisture_param="HS_CV_AVG_-0.2m"  # Proper exclusion!
    )

    print(f"\n   Selected {len(filtered_params)} parameters with >{coverage_threshold*100}% coverage")
    print(f"   Parameters: {sorted(filtered_params)[:10]}...")
    if len(filtered_params) > 10:
        print(f"   ... and {len(filtered_params) - 10} more")

    # Create dataset WITHOUT precomputed data (will build from scratch)
    print("\n3. Creating dataset from timeseries...")
    print("   This builds the sample index (which sequences are valid)")

    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=seq_length,
        n_nearest=n_nearest,
        feature_params=filtered_params,
        precomputed_path=None,  # Force building from scratch
        normalize=False  # We'll normalize during precomputation
    )

    print(f"   ✓ Built index: {len(dataset.sample_index)} valid sequences")

    # Precompute and save with normalization
    print("\n4. Precomputing all sequences with normalization...")
    print("   This is the slow part (~24 hours)")
    print("   Grab a coffee... or 50 ☕")

    output_path = Path(data_dir) / "precomputed_sequences.npz"
    norm_stats_path = Path(data_dir) / "normalization_stats.npz"

    dataset.precompute_and_save(
        output_path=str(output_path),
        norm_stats_path=str(norm_stats_path),
        normalize=True  # Normalize during precomputation
    )

    print("\n" + "=" * 70)
    print("✓ REGENERATION COMPLETE!")
    print("=" * 70)
    print(f"✓ Precomputed sequences: {output_path}")
    print(f"✓ Normalization stats: {norm_stats_path}")
    print(f"✓ Total sequences: {len(dataset.sample_index)}")
    print(f"\nNext steps:")
    print(f"1. Test loading speed:")
    print(f"     python -c \"from Moisturizer import *; ds = SoilMoistureSequenceDataset(...); print(ds[0])\"")
    print(f"2. Start training:")
    print(f"     python your_training_script.py")
    print(f"3. Expect 7,000-10,000 samples/sec with GPU sharing")
    print(f"   Expect 15,000+ samples/sec when running solo")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description='Regenerate dataset from scratch with proper normalization'
    )
    parser.add_argument(
        '--coverage-threshold',
        type=float,
        default=0.25,
        help='Minimum parameter coverage (0.0-1.0, default: 0.25)'
    )
    parser.add_argument(
        '--seq-length',
        type=int,
        default=64,
        help='Sequence length in days (default: 64)'
    )
    parser.add_argument(
        '--n-nearest',
        type=int,
        default=4,
        help='Number of nearest stations (default: 4)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./meteogalicia_data',
        help='Data directory path (default: ./meteogalicia_data)'
    )

    args = parser.parse_args()

    # Confirm before starting
    print("\n⚠  WARNING: This will take approximately 24 hours!")
    print(f"   Data directory: {args.data_dir}")
    print(f"   Coverage threshold: {args.coverage_threshold * 100}%")
    print(f"   Sequence length: {args.seq_length} days")
    print(f"   Nearest stations: {args.n_nearest}")

    response = input("\nProceed with regeneration? [y/N]: ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled.")
        return

    regenerate_dataset(
        coverage_threshold=args.coverage_threshold,
        seq_length=args.seq_length,
        n_nearest=args.n_nearest,
        data_dir=args.data_dir
    )

if __name__ == "__main__":
    main()
