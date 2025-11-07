#!/usr/bin/env python3
"""
Pre-compute augmented dataset with ALL skip patterns and permutations

Strategy:
1. Load base dataset with n_nearest=5 (to get 5th station)
2. For each base sample:
   - For each of 5 skip patterns (skip station 0,1,2,3,4):
     - For each of 24 permutations:
       - Build augmented sample
       - Add to precomputed array
3. Save to precomputed_sequences_augmented.npz

Total: base_samples × 5 skips × 24 perms = base_samples × 120

This takes ~10-20 min to compute once, then training is as fast as base dataset!
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset


def generate_all_augmentations(
    data_dir: str = "./meteogalicia_data",
    n_nearby_available: int = 5,  # Use 5 stations to generate augmentations
    n_nearby_in_features: int = 4,  # But only use 4 in each sample
    coverage_threshold: float = 0.25
):
    """
    Pre-compute ALL augmented samples (deterministic, complete coverage)

    Args:
        data_dir: Data directory
        n_nearby_available: Number of nearby stations to load (5 = original 4 + 1 for skipping)
        n_nearby_in_features: Number to actually use in features (4)
        coverage_threshold: Parameter coverage threshold
    """
    print("=" * 70)
    print("PRE-COMPUTING AUGMENTED DATASET (ALL COMBINATIONS)")
    print("=" * 70)

    collector = MeteoGaliciaCollector(data_dir=data_dir)

    # Get filtered parameters
    print("\n1. Analyzing parameter coverage...")
    coverage, filtered_params = collector.analyze_parameter_coverage(
        coverage_threshold=coverage_threshold
    )

    print(f"   Selected {len(filtered_params)} parameters")

    # Load base dataset with n_nearest=5 (need 5th station for skip augmentation)
    print(f"\n2. Loading base dataset with {n_nearby_available} nearby stations...")

    dense_path = Path(data_dir) / "dense_features.npz"

    base_dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=n_nearby_available,  # Load 5 stations
        feature_params=filtered_params,
        precomputed_path=None,  # Build from scratch
        dense_array_path=str(dense_path) if dense_path.exists() else None,
        normalize=False  # We'll normalize the augmented dataset at the end
    )

    print(f"   Base dataset: {len(base_dataset.sample_index)} samples")

    # Generate all skip patterns and permutations
    print(f"\n3. Generating augmentation combinations...")

    # All ways to pick n_nearby_in_features stations from n_nearby_available
    available_indices = list(range(n_nearby_available))

    # Skip patterns: which station to skip (or keep all if n_nearby_in_features == n_nearby_available)
    skip_patterns = []
    for skip_idx in range(n_nearby_available):
        # Indices to keep (all except skip_idx)
        keep_indices = [i for i in available_indices if i != skip_idx][:n_nearby_in_features]
        skip_patterns.append(keep_indices)

    print(f"   Skip patterns: {len(skip_patterns)}")
    for i, pattern in enumerate(skip_patterns):
        print(f"     Pattern {i}: Skip station {i}, use stations {pattern}")

    # All permutations of n_nearby_in_features stations
    all_permutations = list(permutations(range(n_nearby_in_features)))
    print(f"   Permutations per skip pattern: {len(all_permutations)}")

    total_augmentations = len(skip_patterns) * len(all_permutations)
    print(f"   Total augmentations per base sample: {total_augmentations}")
    print(f"   Total samples in augmented dataset: {len(base_dataset.sample_index) * total_augmentations:,}")

    # Pre-compute all augmented samples
    print(f"\n4. Building all augmented samples...")

    # Get dimensions from first sample
    sample0 = base_dataset[0]
    seq_length = sample0['features'].shape[0]

    # Calculate feature dimensions for n_nearby_in_features=4
    target_features = len(filtered_params)
    nearby_features_per_station = 1 + len(filtered_params) + 1  # distance + features + soil
    total_features = target_features + (nearby_features_per_station * n_nearby_in_features)

    print(f"   Sample shape: [{seq_length}, {total_features}]")

    # Preallocate arrays
    total_samples = len(base_dataset.sample_index) * total_augmentations
    all_features = np.zeros((total_samples, seq_length, total_features), dtype=np.float32)
    all_targets = np.zeros((total_samples, 1), dtype=np.float32)
    all_masks = np.zeros((total_samples, seq_length, total_features), dtype=np.float32)

    # Metadata for each augmented sample
    aug_target_stations = []
    aug_end_dates = []
    aug_start_dates = []
    aug_skip_pattern = []  # Which skip pattern was used
    aug_permutation = []   # Which permutation was used

    aug_idx = 0

    for base_idx in range(len(base_dataset.sample_index)):
        if base_idx % 1000 == 0:
            print(f"   Progress: {base_idx}/{len(base_dataset.sample_index)} ({100*base_idx/len(base_dataset.sample_index):.1f}%)")

        # Get base sample (with 5 nearby stations)
        sample = base_dataset[base_idx]
        base_features = sample['features'].numpy()  # [seq_length, total_features_with_5_stations]
        base_mask = sample['mask'].numpy()
        base_target = sample['target'].numpy()

        # Extract target station features (unchanged)
        target_feat = base_features[:, :target_features]
        target_mask = base_mask[:, :target_features]

        # Extract nearby station features (5 stations)
        nearby_start = target_features
        nearby_features_5 = base_features[:, nearby_start:]  # All 5 stations
        nearby_mask_5 = base_mask[:, nearby_start:]

        # Reshape to separate stations [seq_length, 5, features_per_station]
        nearby_features_5 = nearby_features_5.reshape(
            seq_length,
            n_nearby_available,
            nearby_features_per_station
        )
        nearby_mask_5 = nearby_mask_5.reshape(
            seq_length,
            n_nearby_available,
            nearby_features_per_station
        )

        # For each skip pattern
        for skip_idx, keep_indices in enumerate(skip_patterns):
            # Select 4 stations according to skip pattern
            nearby_features_4 = nearby_features_5[:, keep_indices, :]  # [seq_length, 4, features_per_station]
            nearby_mask_4 = nearby_mask_5[:, keep_indices, :]

            # For each permutation
            for perm_idx, perm in enumerate(all_permutations):
                # Apply permutation to the 4 selected stations
                perm_nearby_features = nearby_features_4[:, perm, :]
                perm_nearby_mask = nearby_mask_4[:, perm, :]

                # Flatten back to feature vector
                perm_nearby_features = perm_nearby_features.reshape(seq_length, -1)
                perm_nearby_mask = perm_nearby_mask.reshape(seq_length, -1)

                # Concatenate target + nearby
                aug_features = np.concatenate([target_feat, perm_nearby_features], axis=1)
                aug_mask = np.concatenate([target_mask, perm_nearby_mask], axis=1)

                # Store augmented sample
                all_features[aug_idx] = aug_features
                all_targets[aug_idx] = base_target
                all_masks[aug_idx] = aug_mask

                # Store metadata
                sample_info = base_dataset.sample_index[base_idx]
                aug_target_stations.append(sample_info['target_station'])
                aug_end_dates.append(sample_info['end_date'].timestamp())
                aug_start_dates.append(sample_info['start_date'].timestamp())
                aug_skip_pattern.append(skip_idx)
                aug_permutation.append(perm_idx)

                aug_idx += 1

    print(f"   Generated {aug_idx:,} augmented samples")

    # Normalize the augmented dataset
    print(f"\n5. Computing normalization statistics...")

    # Use same normalization logic as base dataset
    from Moisturizer import SoilMoistureSequenceDataset

    # Temporarily create a mock dataset to use its normalization methods
    # We'll compute stats from the augmented data directly
    n_features = all_features.shape[2]
    feature_mins = np.full(n_features, np.inf, dtype=np.float32)
    feature_maxs = np.full(n_features, -np.inf, dtype=np.float32)

    invalid_markers = [-9999.0, -1000.0]

    # Compute min/max for each feature
    batch_size = 1000
    for i in range(0, len(all_features), batch_size):
        end_i = min(i + batch_size, len(all_features))
        features_batch = all_features[i:end_i]
        masks_batch = all_masks[i:end_i]

        for feat_idx in range(n_features):
            feat_data = features_batch[:, :, feat_idx]
            feat_mask = masks_batch[:, :, feat_idx]

            # Get valid data
            valid_mask = feat_mask > 0
            for marker in invalid_markers:
                valid_mask &= (feat_data != marker)

            valid_data = feat_data[valid_mask]

            if len(valid_data) > 0:
                feature_mins[feat_idx] = min(feature_mins[feat_idx], valid_data.min())
                feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid_data.max())

    # Compute for targets
    valid_targets = all_targets.copy()
    for marker in invalid_markers:
        valid_targets = valid_targets[valid_targets != marker]

    target_min = valid_targets.min() if len(valid_targets) > 0 else 0.0
    target_max = valid_targets.max() if len(valid_targets) > 0 else 1.0

    print(f"   Feature range: [{feature_mins.min():.2f}, {feature_maxs.max():.2f}]")
    print(f"   Target range: [{target_min:.2f}, {target_max:.2f}]")

    # Apply normalization
    print(f"\n6. Normalizing augmented samples...")

    normalized_invalid_marker = -2.0

    for idx in range(len(all_features)):
        if idx % 10000 == 0:
            print(f"   Progress: {idx}/{len(all_features)} ({100*idx/len(all_features):.1f}%)")

        # Normalize features
        for feat_idx in range(n_features):
            feat_min = feature_mins[feat_idx]
            feat_max = feature_maxs[feat_idx]

            # Handle invalid markers
            invalid_mask = np.zeros(all_features[idx].shape[0], dtype=bool)
            for marker in invalid_markers:
                invalid_mask |= (all_features[idx][:, feat_idx] == marker)

            # Normalize valid values to [-1, 1]
            if feat_max > feat_min:
                all_features[idx][:, feat_idx] = 2.0 * (all_features[idx][:, feat_idx] - feat_min) / (feat_max - feat_min) - 1.0

            # Set invalid markers to -2
            all_features[idx][invalid_mask, feat_idx] = normalized_invalid_marker

        # Normalize target
        target_invalid = False
        for marker in invalid_markers:
            if np.any(all_targets[idx] == marker):
                target_invalid = True
                break

        if target_invalid:
            all_targets[idx][:] = normalized_invalid_marker
        elif target_max > target_min:
            all_targets[idx][:] = 2.0 * (all_targets[idx] - target_min) / (target_max - target_min) - 1.0

    # Save augmented dataset
    print(f"\n7. Saving augmented dataset...")

    output_path = Path(data_dir) / "precomputed_sequences_augmented.npz"
    norm_stats_path = Path(data_dir) / "normalization_stats_augmented.npz"

    np.savez_compressed(
        output_path,
        features=all_features,
        targets=all_targets,
        masks=all_masks,
        target_stations=np.array(aug_target_stations, dtype=np.int32),
        end_dates=np.array(aug_end_dates, dtype=np.float64),
        start_dates=np.array(aug_start_dates, dtype=np.float64),
        skip_pattern=np.array(aug_skip_pattern, dtype=np.int32),
        permutation=np.array(aug_permutation, dtype=np.int32),
        is_normalized=np.array([True], dtype=bool)
    )

    np.savez(
        norm_stats_path,
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
        target_min=target_min,
        target_max=target_max
    )

    print(f"   Saved to: {output_path}")
    print(f"   Normalization stats: {norm_stats_path}")
    print(f"   File size: {output_path.stat().st_size / 1e9:.2f} GB")

    print("\n" + "=" * 70)
    print("✓ AUGMENTED DATASET COMPLETE!")
    print("=" * 70)
    print(f"Base samples: {len(base_dataset.sample_index):,}")
    print(f"Augmented samples: {len(all_features):,}")
    print(f"Augmentation factor: {len(all_features) / len(base_dataset.sample_index):.0f}x")
    print(f"\nBreakdown:")
    print(f"  - Skip patterns: {len(skip_patterns)}")
    print(f"  - Permutations per skip: {len(all_permutations)}")
    print(f"  - Total per base: {total_augmentations}")
    print("\nTo use in training:")
    print("  dataset = SoilMoistureSequenceDataset(...,")
    print("      precomputed_path='precomputed_sequences_augmented.npz',")
    print("      norm_stats_path='normalization_stats_augmented.npz')")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_augmentations()
