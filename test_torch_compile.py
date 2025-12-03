#!/usr/bin/env python3
"""
Benchmark torch.compile optimized AugmentedLiveDataset

Includes proper warmup to account for compilation overhead.
"""

import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from augmented_live_torch import AugmentedLiveDatasetTorch
from augmented_live import AugmentedLiveDataset

print("="*70)
print("TORCH.COMPILE BENCHMARK")
print("="*70)

# Load feature params
dense_data = np.load("meteogalicia_data/dense_features.npz")
feature_params = [p for p in dense_data['feature_params'].tolist() if p != 'HS_CV_AVG_-0.2m']

# Create torch.compile version
print("\n" + "="*70)
print("Creating TORCH.COMPILE dataset...")
print("="*70)
torch_dataset = AugmentedLiveDatasetTorch(
    timeseries="meteogalicia_data/raw_timeseries.csv",
    stations="meteogalicia_data/stations_metadata.csv",
    nearest="meteogalicia_data/nearest_stations.csv",
    seq_length=2,
    n_nearby_available=5,
    n_nearby_in_features=4,
    feature_params=feature_params,
    dense_array_path="meteogalicia_data/dense_features.npz"
)

print(f"\nDataset created: {len(torch_dataset):,} samples")

# WARMUP: Critical for torch.compile!
print("\n" + "="*70)
print("WARMUP PHASE (triggering compilation)")
print("="*70)
print("This will compile the augmentation function on first calls...")

warmup_start = time.time()

# Access samples directly to trigger compilation
print("\n1. Direct sample access warmup (100 samples)...")
for i in range(100):
    _ = torch_dataset[i]
warmup_direct = time.time() - warmup_start
print(f"   Warmup completed in {warmup_direct:.2f}s")

# DataLoader warmup
print("\n2. DataLoader warmup...")
warmup_loader = DataLoader(torch_dataset, batch_size=512, shuffle=False, num_workers=0)
warmup_iter_start = time.time()
for i, batch in enumerate(warmup_loader):
    if i >= 10:
        break
warmup_loader_time = time.time() - warmup_iter_start
print(f"   DataLoader warmup completed in {warmup_loader_time:.2f}s")

print(f"\n✓ Total warmup time: {time.time() - warmup_start:.2f}s")
print("   → Compiled function is now ready for benchmarking")

# Benchmark with different worker counts
print("\n" + "="*70)
print("THROUGHPUT BENCHMARK (after warmup)")
print("="*70)

configs = [
    (512, 0, "Single process"),
    (512, 2, "2 workers"),
    (512, 8, "8 workers"),
]

for batch_size, num_workers, desc in configs:
    print(f"\n{desc} (batch={batch_size}, workers={num_workers}):")

    loader = DataLoader(
        torch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        prefetch_factor=2 if num_workers > 0 else None
    )

    # Additional warmup for this specific config
    for i, batch in enumerate(loader):
        if i >= 2:
            break

    # Measure
    start = time.time()
    samples_processed = 0
    batches = 0

    for batch in loader:
        samples_processed += batch['features'].shape[0]
        batches += 1
        if batches >= 50:
            break

    elapsed = time.time() - start
    throughput = samples_processed / elapsed if elapsed > 0 else 0

    print(f"  {samples_processed:,} samples in {elapsed:.2f}s = {throughput:,.0f} samples/s")

# Compare with numpy version
print("\n" + "="*70)
print("COMPARISON WITH NUMPY VERSION")
print("="*70)

print("\nCreating numpy version for comparison...")
numpy_dataset = AugmentedLiveDataset(
    timeseries="meteogalicia_data/raw_timeseries.csv",
    stations="meteogalicia_data/stations_metadata.csv",
    nearest="meteogalicia_data/nearest_stations.csv",
    seq_length=2,
    n_nearby_available=5,
    n_nearby_in_features=4,
    feature_params=feature_params,
    dense_array_path="meteogalicia_data/dense_features.npz"
)

print("\nNumpy version (8 workers):")
loader = DataLoader(numpy_dataset, batch_size=512, shuffle=False, num_workers=8)

# Warmup
for i, batch in enumerate(loader):
    if i >= 2:
        break

# Measure
start = time.time()
samples_processed = 0
batches = 0

for batch in loader:
    samples_processed += batch['features'].shape[0]
    batches += 1
    if batches >= 50:
        break

elapsed = time.time() - start
numpy_throughput = samples_processed / elapsed if elapsed > 0 else 0

print(f"  {samples_processed:,} samples in {elapsed:.2f}s = {numpy_throughput:,.0f} samples/s")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nExpected baseline (from previous tests):")
print("  Numpy runtime norm:  ~15,000 samples/s (8 workers)")
print("  No normalization:    ~16,500 samples/s (8 workers)")
print("  Precomputed:         ~30,000 samples/s (8 workers)")
print("\nIf torch.compile achieves:")
print("  >20k samples/s → Significant improvement!")
print("  >25k samples/s → Approaching precomputed speed!")
print("  >30k samples/s → Match or beat precomputed!")
print("="*70)
