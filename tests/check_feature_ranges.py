#!/usr/bin/env python3
"""
Check feature ranges per-parameter to identify which features are missing.
"""
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from pathlib import Path
import numpy as np

print("="*70)
print("PER-FEATURE ANALYSIS: BASE vs AUGMENTED")
print("="*70)

# Load base dataset
collector = MeteoGaliciaCollector(data_dir='./meteogalicia_data')
coverage, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
dense_path = Path('./meteogalicia_data') / 'dense_features.npz'

print("\nFiltered params (in order):")
for i, param in enumerate(filtered_params):
    print(f"  {i}: {param}")
print()

base_dataset = SoilMoistureSequenceDataset(
    timeseries=str(collector.timeseries_file),
    stations=str(collector.stations_file),
    nearest=str(collector.nearest_file),
    seq_length=64,
    n_nearest=5,
    feature_params=filtered_params,
    precomputed_path=None,
    dense_array_path=str(dense_path),
    normalize=False
)

print(f"\nBase dataset: {len(base_dataset)} samples")
print()

# Sample 100 random samples from base dataset
num_samples = min(100, len(base_dataset))
indices = np.random.choice(len(base_dataset), size=num_samples, replace=False)

print(f"Sampling {num_samples} samples from base dataset...")
base_features = []
for idx in indices:
    sample = base_dataset[int(idx)]
    base_features.append(sample['features'].numpy())

base_features = np.stack(base_features, axis=0)  # (100, 64, 166)
print(f"Base features shape: {base_features.shape}")
print()

# Get feature names
feature_names = base_dataset.get_feature_names()
print(f"Total features: {len(feature_names)}")
print()

# Compute per-feature stats for base dataset
print("="*70)
print("BASE DATASET PER-FEATURE STATISTICS (target station only)")
print("="*70)

target_features_count = len(filtered_params)
print(f"\nTarget station has {target_features_count} features")
print()

for feat_idx in range(target_features_count):
    feat_name = feature_names[feat_idx]
    feat_values = base_features[:, :, feat_idx].flatten()
    valid_values = feat_values[feat_values != -9999.0]

    if len(valid_values) > 0:
        feat_min = valid_values.min()
        feat_max = valid_values.max()
        print(f"{feat_idx:2d}. {feat_name:40s} [{feat_min:12.2f}, {feat_max:12.2f}]")
    else:
        print(f"{feat_idx:2d}. {feat_name:40s} [NO VALID DATA]")

# Now check augmented dataset
aug_path = Path('./meteogalicia_data') / 'precomputed_sequences_augmented'
if aug_path.exists():
    print()
    print("="*70)
    print("AUGMENTED DATASET PER-FEATURE STATISTICS (target station only)")
    print("="*70)

    # Load memmap
    features_path = aug_path / 'features.npy'
    if features_path.exists():
        aug_features = np.lib.format.open_memmap(str(features_path), mode='r')
        print(f"\nAugmented features shape: {aug_features.shape}")

        # Sample 100 random samples
        num_aug_samples = min(100, len(aug_features))
        aug_indices = np.random.choice(len(aug_features), size=num_aug_samples, replace=False)

        aug_sample_features = aug_features[aug_indices]  # (100, 64, 138)
        print(f"Sampled {num_aug_samples} augmented samples")
        print()

        # Target features in augmented dataset (same count)
        aug_target_features_count = target_features_count  # Should be same

        for feat_idx in range(aug_target_features_count):
            feat_name = filtered_params[feat_idx]
            feat_values = aug_sample_features[:, :, feat_idx].flatten()
            valid_values = feat_values[feat_values != -9999.0]

            if len(valid_values) > 0:
                feat_min = valid_values.min()
                feat_max = valid_values.max()
                print(f"{feat_idx:2d}. target_{feat_name:35s} [{feat_min:12.2f}, {feat_max:12.2f}]")
            else:
                print(f"{feat_idx:2d}. target_{feat_name:35s} [NO VALID DATA]")

        # Check nearby station features too
        print()
        print("="*70)
        print("AUGMENTED DATASET: NEARBY STATION DISTANCE FEATURE")
        print("="*70)

        for nearby_idx in range(4):  # 4 nearby stations
            offset = target_features_count + (nearby_idx * 28)  # 28 = 1 + 26 + 1
            distance_feat_idx = offset

            dist_values = aug_sample_features[:, :, distance_feat_idx].flatten()
            valid_dist = dist_values[dist_values != -9999.0]

            if len(valid_dist) > 0:
                print(f"Nearby {nearby_idx+1} distance: [{valid_dist.min():12.2f}, {valid_dist.max():12.2f}]")
            else:
                print(f"Nearby {nearby_idx+1} distance: [NO VALID DATA]")
else:
    print("\n❌ Augmented dataset not found")
