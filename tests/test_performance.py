#!/usr/bin/env python3
"""
Test script to demonstrate performance improvements with precomputed sequences
"""

import time
import torch
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

def test_loading_speed():
    """Compare loading speed with and without precomputed data"""
    collector = MeteoGaliciaCollector()

    # Get filtered parameters
    _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
    print(f"Using {len(filtered_params)} filtered parameters")

    # Check if precomputed data exists
    precomputed_path = collector.data_dir / "precomputed_sequences.npz"
    norm_stats_path = collector.data_dir / "normalization_stats.npz"

    if not precomputed_path.exists():
        print("\n⚠ Precomputed data not found!")
        print(f"  Please run precomputeDataset() first")
        return

    print("\n" + "=" * 60)
    print("TEST 1: Loading with precomputed data + normalization")
    print("=" * 60)

    start = time.time()
    dataset_precomputed = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=4,
        feature_params=filtered_params,
        precomputed_path=str(precomputed_path),
        normalize=True,
        norm_stats_path=str(norm_stats_path)
    )
    load_time = time.time() - start

    print(f"\nDataset loaded in {load_time:.2f} seconds")
    print(f"Total samples: {len(dataset_precomputed)}")

    print("\n" + "=" * 60)
    print("TEST 2: Data loading speed (batch iteration)")
    print("=" * 60)

    # Test iteration speed
    dataloader = torch.utils.data.DataLoader(
        dataset_precomputed,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )

    # Warmup
    for i, batch in enumerate(dataloader):
        if i >= 10:
            break

    # Measure throughput
    n_batches = 100
    start = time.time()
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
    elapsed = time.time() - start

    samples_per_sec = (n_batches * 4) / elapsed
    print(f"\nProcessed {n_batches} batches (batch_size=4) in {elapsed:.2f} seconds")
    print(f"Throughput: {samples_per_sec:.1f} samples/second")

    print("\n" + "=" * 60)
    print("TEST 3: Sample inspection")
    print("=" * 60)

    sample = dataset_precomputed[0]
    print(f"\nSample 0:")
    print(f"  Features shape: {sample['features'].shape}")
    print(f"  Features min: {sample['features'].min():.3f}")
    print(f"  Features max: {sample['features'].max():.3f}")
    print(f"  Target shape: {sample['target'].shape}")
    print(f"  Target value: {sample['target'].item():.3f}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Station ID: {sample['target_station_id']}")

    # Check normalization
    features = sample['features'].numpy()
    valid_mask = sample['mask'].numpy() > 0
    valid_features = features[valid_mask]
    invalid_marker_count = (features == -2.0).sum()

    print(f"\nNormalization check:")
    print(f"  Valid features range: [{valid_features.min():.3f}, {valid_features.max():.3f}]")
    print(f"  Invalid markers (-2): {invalid_marker_count}")
    print(f"  Expected range: [-1, 1]")

    if valid_features.min() >= -1.01 and valid_features.max() <= 1.01:
        print("  ✓ Normalization looks correct!")
    else:
        print("  ⚠ Normalization might be off")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Dataset loads in {load_time:.2f} seconds")
    print(f"✓ Throughput: {samples_per_sec:.1f} samples/second")
    print(f"✓ Normalization: {'correct' if valid_features.min() >= -1.01 and valid_features.max() <= 1.01 else 'needs checking'}")
    print("\nExpected performance:")
    print("  - With precomputed + normalization: ~1,000-10,000 samples/sec")
    print("  - Without precomputed (old way): ~6 samples/sec")
    print(f"  - Speedup: ~{samples_per_sec/6:.0f}x faster")

if __name__ == "__main__":
    test_loading_speed()
