#!/usr/bin/env python3
"""
Test AugmentedLiveDataset with tiny dataset (seq_length=2)
Compare against precomputed augmented version for correctness
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from augmented_live import AugmentedLiveDataset
from Moisturizer import SoilMoistureSequenceDataset

print("="*70)
print("TESTING AugmentedLiveDataset (seq_length=2)")
print("="*70)

# Get feature params from dense arrays
import numpy as np
dense_data = np.load("meteogalicia_data/dense_features.npz")
feature_params = dense_data['feature_params'].tolist()
# Remove soil moisture and coordinates - those are handled separately
feature_params = [p for p in feature_params if p not in ['HS_CV_AVG_-0.2m', 'altitude', 'utmx', 'utmy']]
print(f"Using {len(feature_params)} feature parameters from dense arrays")

# Create live augmented dataset
print("\nCreating LIVE augmented dataset...")
live_dataset = AugmentedLiveDataset(
    timeseries="meteogalicia_data/raw_timeseries.csv",
    stations="meteogalicia_data/stations_metadata.csv",
    nearest="meteogalicia_data/nearest_stations.csv",
    seq_length=2,  # Tiny for testing
    n_nearby_available=5,
    n_nearby_in_features=4,
    feature_params=feature_params,
    dense_array_path="meteogalicia_data/dense_features.npz"
)

print(f"\nLive dataset: {len(live_dataset):,} samples")

# Load precomputed augmented dataset for comparison
print("\nLoading PRECOMPUTED augmented dataset...")
precomputed_dataset = SoilMoistureSequenceDataset(
    timeseries="meteogalicia_data/raw_timeseries.csv",
    stations="meteogalicia_data/stations_metadata.csv",
    nearest="meteogalicia_data/nearest_stations.csv",
    seq_length=2,
    n_nearest=4,
    feature_params=None,  # Load with whatever params it was precomputed with
    precomputed_path="meteogalicia_data/precomputed_sequences_augmented",
    normalize=False
)

print(f"Precomputed dataset: {len(precomputed_dataset):,} samples")

if len(live_dataset) == 0:
    print("\n✗ Live dataset is empty! Need more days of data.")
    exit(1)

print(f"\n{'='*70}")
print("CORRECTNESS TESTS")
print("="*70)

# Compare a few samples
print("\n1. Comparing live vs precomputed samples...")
num_compare = min(10, len(live_dataset), len(precomputed_dataset))

for i in [0, 1, 5, num_compare-1]:
    if i >= len(live_dataset) or i >= len(precomputed_dataset):
        continue

    live_sample = live_dataset[i]
    precomp_sample = precomputed_dataset[i]

    live_feat = live_sample['features'].numpy()
    precomp_feat = precomp_sample['features'].numpy()

    live_target = live_sample['target'].item()
    precomp_target = precomp_sample['target'].item()

    print(f"\n   Sample {i}:")
    print(f"     Live features shape: {live_feat.shape}")
    print(f"     Precomp features shape: {precomp_feat.shape}")

    # Check if shapes match
    if live_feat.shape != precomp_feat.shape:
        print(f"     ✗ Shape mismatch! Live: {live_feat.shape}, Precomp: {precomp_feat.shape}")
        continue

    # Check feature values (allowing small numerical differences)
    feat_diff = np.abs(live_feat - precomp_feat)
    max_diff = feat_diff.max()
    mean_diff = feat_diff.mean()

    print(f"     Feature diff: max={max_diff:.6f}, mean={mean_diff:.6f}")

    # Check target
    target_diff = abs(live_target - precomp_target)
    print(f"     Target diff: {target_diff:.6f}")
    print(f"     Live target: {live_target:.4f}, Precomp target: {precomp_target:.4f}")

    if max_diff < 1e-4 and target_diff < 1e-4:
        print(f"     ✓ Match!")
    else:
        print(f"     ⚠ Differences detected (may be due to different augmentation order)")

print("\n2. Testing sample statistics...")
# Get a batch of samples
batch_size = 100
num_samples = min(batch_size, len(live_dataset))

live_samples = [live_dataset[i] for i in range(num_samples)]
live_features = torch.stack([s['features'] for s in live_samples]).numpy()
live_targets = torch.stack([s['target'] for s in live_samples]).numpy()

print(f"\n   Analyzed {num_samples} samples:")
print(f"   Feature range: [{live_features.min():.4f}, {live_features.max():.4f}]")
print(f"   Target range: [{live_targets.min():.4f}, {live_targets.max():.4f}]")

# Check normalized range
valid_features = live_features[live_features != -2.0]
if len(valid_features) > 0:
    if valid_features.min() >= -1.1 and valid_features.max() <= 1.1:
        print(f"   ✓ Features in normalized range [-1, 1]")
    else:
        print(f"   ⚠ Features outside normalized range")

valid_targets = live_targets[live_targets != -2.0]
if len(valid_targets) > 0:
    if valid_targets.min() >= -1.1 and valid_targets.max() <= 1.1:
        print(f"   ✓ Targets in normalized range [-1, 1]")
    else:
        print(f"   ⚠ Targets outside normalized range")

print(f"\n{'='*70}")
print("THROUGHPUT TESTS")
print("="*70)

# Test throughput with different configurations
print("\n3. Testing DataLoader throughput...")

configs = [
    (128, 0),   # Single process
    (128, 2),   # 2 workers
    (512, 2),   # Larger batch
]

for batch_size, num_workers in configs:
    print(f"\n   Batch size: {batch_size}, Workers: {num_workers}")

    # Test LIVE dataset
    print(f"     LIVE dataset:")
    live_loader = DataLoader(
        live_dataset,
        batch_size=batch_size,
        shuffle=False,  # Sequential for fair comparison
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Warm up
    for i, batch in enumerate(live_loader):
        if i >= 2:
            break

    # Measure
    start = time.time()
    samples_processed = 0
    batches = 0

    for batch in live_loader:
        samples_processed += batch['features'].shape[0]
        batches += 1
        if batches >= 50:  # Limit batches
            break

    elapsed = time.time() - start
    live_throughput = samples_processed / elapsed if elapsed > 0 else 0

    print(f"       {samples_processed} samples in {elapsed:.2f}s = {live_throughput:.1f} samples/s")

    # Test PRECOMPUTED dataset
    print(f"     PRECOMPUTED dataset:")
    precomp_loader = DataLoader(
        precomputed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Warm up
    for i, batch in enumerate(precomp_loader):
        if i >= 2:
            break

    # Measure
    start = time.time()
    samples_processed = 0
    batches = 0

    for batch in precomp_loader:
        samples_processed += batch['features'].shape[0]
        batches += 1
        if batches >= 50:
            break

    elapsed = time.time() - start
    precomp_throughput = samples_processed / elapsed if elapsed > 0 else 0

    print(f"       {samples_processed} samples in {elapsed:.2f}s = {precomp_throughput:.1f} samples/s")

    # Compare
    if precomp_throughput > 0:
        ratio = live_throughput / precomp_throughput
        print(f"     Ratio: {ratio:.2f}x (live vs precomputed)")
        if ratio > 0.8:
            print(f"     ✓ Live is within 80% of precomputed speed")
        else:
            print(f"     ⚠ Live is slower than expected")

print(f"\n{'='*70}")
print("✓ TESTS COMPLETE!")
print("="*70)
