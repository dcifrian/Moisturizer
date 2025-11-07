#!/usr/bin/env python3
"""
Regenerate the entire dataset from scratch with proper normalization

This script:
1. Analyzes parameter coverage
2. Builds dense feature arrays (FAST - eliminates redundancy!)
3. Generates sample index for all valid sequences
4. Pre-computes and normalizes all sequences using dense arrays
5. Saves everything with correct normalization stats

Run this once, wait ~2-5 minutes, then enjoy 10,000+ samples/sec training!
"""

import argparse
import subprocess
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
    print("REGENERATING DATASET FROM SCRATCH (OPTIMIZED!)")
    print("=" * 70)
    print(f"With dense arrays: ~2-5 minutes instead of 24 hours!")
    print("=" * 70)

    # Initialize collector
    print("\n1. Initializing data collector...")
    collector = MeteoGaliciaCollector(data_dir=data_dir)

    # Build dense arrays first (MASSIVE SPEEDUP!)
    print("\n2. Building dense feature arrays...")
    print("   Running build_dense_features.py...")
    subprocess.run(["python3", "build_dense_features.py"], check=True)

    dense_array_path = Path(data_dir) / "dense_features.npz"
    if not dense_array_path.exists():
        print(f"✗ Error: Dense arrays not created at {dense_array_path}")
        return

    print(f"   ✓ Dense arrays created!")

    # Analyze parameter coverage (already done in build_dense, but get the list)
    print("\n3. Analyzing parameter coverage...")
    coverage, filtered_params = collector.analyze_parameter_coverage(
        coverage_threshold=coverage_threshold,
        soil_moisture_param="HS_CV_AVG_-0.2m"  # Proper exclusion!
    )

    print(f"\n   Selected {len(filtered_params)} parameters with >{coverage_threshold*100}% coverage")

    # Create dataset WITH dense arrays (FAST!)
    print("\n4. Creating dataset with dense arrays...")
    print("   This builds the sample index (which sequences are valid)")

    dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=seq_length,
        n_nearest=n_nearest,
        feature_params=filtered_params,
        precomputed_path=None,  # Will be created
        dense_array_path=str(dense_array_path),  # Use dense arrays!
        normalize=False  # We'll normalize during precomputation
    )

    print(f"   ✓ Built index: {len(dataset.sample_index)} valid sequences")

    # Precompute and save with normalization (FAST with dense arrays!)
    print("\n5. Precomputing all sequences with normalization...")
    print("   With dense arrays this should take ~2-5 minutes!")

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
    print(f"✓ Dense arrays: {dense_array_path}")
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
    print(f"\n🚀 Generation time: ~2-5 minutes (down from 24 hours!)")
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
    print("\n✓  With dense array optimization: ~2-5 minutes!")
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
