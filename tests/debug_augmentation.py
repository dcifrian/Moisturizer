#!/usr/bin/env python3
"""
Minimal debug script to trace the augmentation bug.
Generates just a few batches to check what's being written.
"""
from precompute_augmented import generate_all_augmentations_batched
from pathlib import Path
import shutil

# Clean up any existing augmented data
aug_path = Path("./meteogalicia_data/precomputed_sequences_augmented")
if aug_path.exists():
    print(f"Removing existing augmented dataset...")
    shutil.rmtree(aug_path)

# Generate with just 2 batches for debugging
print("\nGenerating 2 batches for debugging...")
print("=" * 70)

generate_all_augmentations_batched(
    data_dir="./meteogalicia_data",
    n_nearby_available=5,
    n_nearby_in_features=4,
    coverage_threshold=0.25,
    seq_length=64,
    batch_size=2,  # Very small batches for debugging
    num_workers=1,  # Single worker to avoid race conditions
    use_base_stats=False
)

print("\n" + "=" * 70)
print("Debug run complete! Checking results...")
print("=" * 70)

import numpy as np

# Load and check the results
targets = np.load(aug_path / "targets.npy", mmap_mode='r')
features = np.load(aug_path / "features.npy", mmap_mode='r')

print(f"\nTargets shape: {targets.shape}, dtype: {targets.dtype}")
print(f"Features shape: {features.shape}, dtype: {features.dtype}")
print(f"\nFirst 20 target values: {targets[:20].flatten()}")
print(f"Target range: [{targets.min():.6f}, {targets.max():.6f}]")
print(f"Target unique values (first 240): {len(np.unique(targets[:240]))} unique")
print(f"\nFeature range: [{features.min():.6f}, {features.max():.6f}]")
print(f"\nAre ALL targets either 0 or 1?: {np.all((targets == 0) | (targets == 1))}")

# Check if targets look like booleans
bool_like = np.sum((targets == 0) | (targets == 1))
print(f"Number of targets that are 0 or 1: {bool_like} / {len(targets)} ({100*bool_like/len(targets):.1f}%)")
