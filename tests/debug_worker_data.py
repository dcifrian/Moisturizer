#!/usr/bin/env python3
"""
Debug script to see what data is being passed to workers.
Runs with just 1 batch (10 base samples) to capture debug output.
"""
from precompute_augmented import generate_all_augmentations_batched
from pathlib import Path
import shutil

# Clean up existing augmented dataset
aug_path = Path("./meteogalicia_data/precomputed_sequences_augmented")
if aug_path.exists():
    print("Removing existing augmented dataset...")
    shutil.rmtree(aug_path)

print("\n" + "="*70)
print("DEBUG RUN: Processing just 1 batch (10 samples)")
print("="*70)

generate_all_augmentations_batched(
    data_dir="./meteogalicia_data",
    n_nearby_available=5,
    n_nearby_in_features=4,
    coverage_threshold=0.25,
    seq_length=2,  # Match tiny dataset
    batch_size=10,  # Small batch
    num_workers=1,  # Single worker for clear debug output
    use_base_stats=False
)

print("\n" + "="*70)
print("Checking results...")
print("="*70)

import numpy as np

# Check what was written
targets = np.load(aug_path / "targets.npy", mmap_mode='r')
features = np.load(aug_path / "features.npy", mmap_mode='r')

print(f"\nFirst 10 targets written: {targets[:10].flatten()}")
print(f"Target range: [{targets.min():.6f}, {targets.max():.6f}]")
print(f"Feature range: [{features.min():.6f}, {features.max():.6f}]")

# Check for large values (UTM coordinates)
has_large = False
for i in range(features.shape[2]):
    if features[:, :, i].max() > 100000:
        has_large = True
        print(f"  Found large values in feature {i}: max={features[:, :, i].max():.2f}")

if not has_large:
    print("  ❌ NO LARGE VALUES - UTM coordinates missing!")
