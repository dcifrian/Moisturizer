#!/usr/bin/env python3
"""
Check if the base dataset (used for augmentation) has valid targets.
This replicates what the user runs on their machine.
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np

print("="*70)
print("CHECKING BASE DATASET TARGETS")
print("="*70)

# Load dataset exactly as precompute_augmented.py does
collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'

print(f"\nFiltered params: {len(filtered_params)}")
print(f"Dense path exists: {dense_path.exists()}")
print()

# Build base dataset with 5 nearby stations (one more than final augmented)
# This is NOT normalized and does NOT use precomputed
seq_length = 2
n_nearby_available = 5

print(f"Building base dataset (seq_length={seq_length}, n_nearest={n_nearby_available})...")
base_dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=seq_length,
    n_nearest=n_nearby_available,
    feature_params=filtered_params,
    precomputed_path=None,  # NOT using precomputed
    dense_array_path=str(dense_path),
    normalize=False  # NOT normalized
)

print(f"Dataset has {len(base_dataset)} samples")
print()

# Check first sample
if len(base_dataset) > 0:
    print("Checking first sample...")
    sample = base_dataset[0]
    target = sample['target'].numpy()
    features = sample['features'].numpy()

    print(f"  Target: {target} (shape={target.shape}, dtype={target.dtype})")
    print(f"  Features: shape={features.shape}, dtype={features.dtype}")
    print(f"  Feature range: [{features.min():.6f}, {features.max():.6f}]")
    print()

    # Check 10 random samples
    print("Checking 10 random samples...")
    indices = np.random.choice(len(base_dataset), size=min(10, len(base_dataset)), replace=False)
    targets = []
    for idx in indices:
        sample = base_dataset[int(idx)]
        targets.append(sample['target'].numpy()[0])

    targets = np.array(targets)
    print(f"  Target values: {targets}")
    print(f"  Range: [{targets.min():.6f}, {targets.max():.6f}]")

    # Count invalid targets
    invalid_count = np.sum((targets == -1000.0) | (targets == -9999.0))
    valid_count = len(targets) - invalid_count
    print(f"  Valid: {valid_count}/{len(targets)}")
    print(f"  Invalid: {invalid_count}/{len(targets)}")

    if invalid_count == len(targets):
        print("\n❌ ALL TARGETS ARE INVALID!")
    elif invalid_count > 0:
        print(f"\n⚠️  {invalid_count}/{len(targets)} targets are invalid")
    else:
        print("\n✅ All sampled targets are valid")
