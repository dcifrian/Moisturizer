#!/usr/bin/env python3
"""
Check what the base dataset (5 nearby stations, raw data) actually contains
This will help diagnose why the augmented dataset gets corrupted
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np

print("="*70)
print("DIAGNOSING BASE DATASET (5 nearby stations, unnormalized)")
print("="*70)

collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'

# Load exactly as precompute_augmented.py does
base_dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=2,  # Use seq_length=2 for quick test
    n_nearest=5,   # 5 nearby stations (will become 4 in augmented)
    feature_params=filtered_params,
    precomputed_path=None,  # Build from scratch, not precomputed!
    dense_array_path=str(dense_path) if dense_path.exists() else None,
    normalize=False  # No normalization!
)

print(f"\nBase dataset: {len(base_dataset.sample_index)} samples")
print(f"Is normalized: {base_dataset.normalize}")
print()

if len(base_dataset.sample_index) > 0:
    # Get first sample
    sample = base_dataset[0]
    features = sample['features'].numpy()
    target = sample['target'].numpy()
    mask = sample['mask'].numpy()

    print("First sample:")
    print(f"  Features: shape={features.shape}, dtype={features.dtype}")
    print(f"  Target: {target} (shape={target.shape}, dtype={target.dtype})")
    print(f"  Mask: shape={mask.shape}, dtype={mask.dtype}")
    print()

    # Check feature ranges
    print(f"  Feature range: [{features.min():.2f}, {features.max():.2f}]")
    print(f"  Target range: [{target.min():.6f}, {target.max():.6f}]")
    print()

    # Check for large values (UTM coordinates)
    print("  Checking for UTM coordinates (large values > 100000):")
    found_large = False
    for i in range(features.shape[1]):
        col_max = features[:, i].max()
        if col_max > 100000:
            print(f"    Feature {i}: max={col_max:.2f}")
            found_large = True
    if not found_large:
        print("    ❌ NO LARGE VALUES FOUND")
    print()

    # Check if target looks like a boolean
    if target.item() in [0.0, 1.0]:
        print(f"  ⚠️  WARNING: Target is exactly {target.item()} - looks like boolean!")
    else:
        print(f"  ✅ Target looks normal (soil moisture value)")
    print()

    # Check if mask dtype is correct
    print(f"  Mask dtype: {mask.dtype}")
    if mask.dtype == np.float32:
        print(f"  ⚠️  WARNING: Mask is float32, should be bool!")
    elif mask.dtype == bool:
        print(f"  ✅ Mask dtype is correct (bool)")
    print()

    # Sample more targets to see the distribution
    print("  Checking 10 random samples...")
    indices = np.random.choice(len(base_dataset.sample_index), min(10, len(base_dataset.sample_index)), replace=False)
    targets_sample = []
    for idx in indices:
        s = base_dataset[int(idx)]
        targets_sample.append(s['target'].numpy().item())

    targets_sample = np.array(targets_sample)
    print(f"    Target values: {targets_sample}")
    print(f"    Range: [{targets_sample.min():.6f}, {targets_sample.max():.6f}]")
    bool_count = np.sum((targets_sample == 0.0) | (targets_sample == 1.0))
    print(f"    Exactly 0 or 1: {bool_count}/10")

    if bool_count > 5:
        print(f"    ⚠️  More than half are 0 or 1 - something is wrong!")
else:
    print("❌ No samples found in base dataset!")
