#!/usr/bin/env python3
"""
Pre-compute augmented dataset with BATCHED processing (memory efficient!)

Strategy:
1. Process base samples in small batches (e.g., 100 at a time)
2. For each batch, generate all 120 augmentations
3. Save batch to temporary file
4. Merge all batches at the end

Memory usage: Only ~1-2GB instead of 360GB!
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
import tempfile
import shutil


def generate_all_augmentations_batched(
    data_dir: str = "./meteogalicia_data",
    n_nearby_available: int = 5,
    n_nearby_in_features: int = 4,
    coverage_threshold: float = 0.25,
    batch_size: int = 100  # Process 100 base samples at a time
):
    """
    Pre-compute ALL augmented samples with batched processing (memory efficient!)
    """
    print("=" * 70)
    print("PRE-COMPUTING AUGMENTED DATASET (MEMORY EFFICIENT)")
    print("=" * 70)

    collector = MeteoGaliciaCollector(data_dir=data_dir)

    # Get filtered parameters
    print("\n1. Analyzing parameter coverage...")
    coverage, filtered_params = collector.analyze_parameter_coverage(
        coverage_threshold=coverage_threshold
    )

    print(f"   Selected {len(filtered_params)} parameters")

    # Load base dataset with n_nearest=5
    print(f"\n2. Loading base dataset with {n_nearby_available} nearby stations...")

    dense_path = Path(data_dir) / "dense_features.npz"

    base_dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=n_nearby_available,
        feature_params=filtered_params,
        precomputed_path=None,
        dense_array_path=str(dense_path) if dense_path.exists() else None,
        normalize=False
    )

    print(f"   Base dataset: {len(base_dataset.sample_index)} samples")

    # Generate augmentation combinations
    print(f"\n3. Generating augmentation combinations...")

    available_indices = list(range(n_nearby_available))
    skip_patterns = []
    for skip_idx in range(n_nearby_available):
        keep_indices = [i for i in available_indices if i != skip_idx][:n_nearby_in_features]
        skip_patterns.append(keep_indices)

    all_permutations = list(permutations(range(n_nearby_in_features)))
    total_augmentations = len(skip_patterns) * len(all_permutations)

    print(f"   Skip patterns: {len(skip_patterns)}")
    print(f"   Permutations per skip: {len(all_permutations)}")
    print(f"   Total augmentations per base: {total_augmentations}")
    print(f"   Total augmented samples: {len(base_dataset.sample_index) * total_augmentations:,}")

    # Get dimensions
    sample0 = base_dataset[0]
    seq_length = sample0['features'].shape[0]
    target_features = len(filtered_params)
    nearby_features_per_station = 1 + len(filtered_params) + 1
    total_features = target_features + (nearby_features_per_station * n_nearby_in_features)

    print(f"   Sample shape: [{seq_length}, {total_features}]")

    # Create temporary directory for batch files
    temp_dir = Path(tempfile.mkdtemp(prefix="augmented_batches_"))
    print(f"\n4. Processing in batches of {batch_size} (temp dir: {temp_dir})...")

    batch_files = []
    num_batches = (len(base_dataset.sample_index) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(base_dataset.sample_index))
        batch_actual_size = end_idx - start_idx

        print(f"\n   Batch {batch_num+1}/{num_batches}: Processing base samples {start_idx}-{end_idx}...")

        # Allocate arrays for this batch only
        batch_aug_size = batch_actual_size * total_augmentations
        batch_features = np.zeros((batch_aug_size, seq_length, total_features), dtype=np.float32)
        batch_targets = np.zeros((batch_aug_size, 1), dtype=np.float32)
        batch_masks = np.zeros((batch_aug_size, seq_length, total_features), dtype=np.float32)
        batch_target_stations = []
        batch_end_dates = []
        batch_start_dates = []
        batch_skip_pattern = []
        batch_permutation = []

        aug_idx = 0

        for base_idx in range(start_idx, end_idx):
            # Get base sample
            sample = base_dataset[base_idx]
            base_features = sample['features'].numpy()
            base_mask = sample['mask'].numpy()
            base_target = sample['target'].numpy()

            # Extract target and nearby features
            target_feat = base_features[:, :target_features]
            target_mask = base_mask[:, :target_features]

            nearby_start = target_features
            nearby_features_5 = base_features[:, nearby_start:].reshape(
                seq_length, n_nearby_available, nearby_features_per_station
            )
            nearby_mask_5 = base_mask[:, nearby_start:].reshape(
                seq_length, n_nearby_available, nearby_features_per_station
            )

            # Generate all augmentations for this base sample
            for skip_idx, keep_indices in enumerate(skip_patterns):
                nearby_features_4 = nearby_features_5[:, keep_indices, :]
                nearby_mask_4 = nearby_mask_5[:, keep_indices, :]

                for perm_idx, perm in enumerate(all_permutations):
                    perm_nearby_features = nearby_features_4[:, perm, :].reshape(seq_length, -1)
                    perm_nearby_mask = nearby_mask_4[:, perm, :].reshape(seq_length, -1)

                    aug_features = np.concatenate([target_feat, perm_nearby_features], axis=1)
                    aug_mask = np.concatenate([target_mask, perm_nearby_mask], axis=1)

                    batch_features[aug_idx] = aug_features
                    batch_targets[aug_idx] = base_target
                    batch_masks[aug_idx] = aug_mask

                    sample_info = base_dataset.sample_index[base_idx]
                    batch_target_stations.append(sample_info['target_station'])
                    batch_end_dates.append(sample_info['end_date'].timestamp())
                    batch_start_dates.append(sample_info['start_date'].timestamp())
                    batch_skip_pattern.append(skip_idx)
                    batch_permutation.append(perm_idx)

                    aug_idx += 1

        # Save batch to temporary file
        batch_file = temp_dir / f"batch_{batch_num:04d}.npz"
        np.savez_compressed(
            batch_file,
            features=batch_features,
            targets=batch_targets,
            masks=batch_masks,
            target_stations=np.array(batch_target_stations, dtype=np.int32),
            end_dates=np.array(batch_end_dates, dtype=np.float64),
            start_dates=np.array(batch_start_dates, dtype=np.float64),
            skip_pattern=np.array(batch_skip_pattern, dtype=np.int32),
            permutation=np.array(batch_permutation, dtype=np.int32)
        )
        batch_files.append(batch_file)

        print(f"      Saved {aug_idx} augmented samples to {batch_file.name}")
        print(f"      Memory freed for next batch...")

        # Explicitly free memory
        del batch_features, batch_targets, batch_masks
        del batch_target_stations, batch_end_dates, batch_start_dates
        del batch_skip_pattern, batch_permutation

    # Merge all batches
    print(f"\n5. Merging {len(batch_files)} batches...")
    print(f"   This will take a few minutes...")

    # Load all batches and concatenate
    all_features_list = []
    all_targets_list = []
    all_masks_list = []
    all_target_stations = []
    all_end_dates = []
    all_start_dates = []
    all_skip_pattern = []
    all_permutation = []

    for i, batch_file in enumerate(batch_files):
        print(f"   Loading batch {i+1}/{len(batch_files)}...")
        batch_data = np.load(batch_file)

        all_features_list.append(batch_data['features'])
        all_targets_list.append(batch_data['targets'])
        all_masks_list.append(batch_data['masks'])
        all_target_stations.extend(batch_data['target_stations'])
        all_end_dates.extend(batch_data['end_dates'])
        all_start_dates.extend(batch_data['start_dates'])
        all_skip_pattern.extend(batch_data['skip_pattern'])
        all_permutation.extend(batch_data['permutation'])

    print(f"   Concatenating arrays...")
    all_features = np.concatenate(all_features_list, axis=0)
    all_targets = np.concatenate(all_targets_list, axis=0)
    all_masks = np.concatenate(all_masks_list, axis=0)

    print(f"   Total samples: {len(all_features):,}")

    # Normalize
    print(f"\n6. Computing normalization statistics...")

    n_features = all_features.shape[2]
    feature_mins = np.full(n_features, np.inf, dtype=np.float32)
    feature_maxs = np.full(n_features, -np.inf, dtype=np.float32)
    invalid_markers = [-9999.0, -1000.0]

    # Sample-based statistics computation (memory efficient)
    sample_batch_size = 10000
    for i in range(0, len(all_features), sample_batch_size):
        end_i = min(i + sample_batch_size, len(all_features))
        features_batch = all_features[i:end_i]
        masks_batch = all_masks[i:end_i]

        for feat_idx in range(n_features):
            feat_data = features_batch[:, :, feat_idx]
            feat_mask = masks_batch[:, :, feat_idx]

            valid_mask = feat_mask > 0
            for marker in invalid_markers:
                valid_mask &= (feat_data != marker)

            valid_data = feat_data[valid_mask]

            if len(valid_data) > 0:
                feature_mins[feat_idx] = min(feature_mins[feat_idx], valid_data.min())
                feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid_data.max())

    valid_targets = all_targets[~np.isin(all_targets, invalid_markers)]
    target_min = valid_targets.min() if len(valid_targets) > 0 else 0.0
    target_max = valid_targets.max() if len(valid_targets) > 0 else 1.0

    print(f"   Feature range: [{feature_mins.min():.2f}, {feature_maxs.max():.2f}]")
    print(f"   Target range: [{target_min:.2f}, {target_max:.2f}]")

    # Normalize in batches
    print(f"\n7. Normalizing augmented samples...")

    normalized_invalid_marker = -2.0

    for idx in range(0, len(all_features), sample_batch_size):
        end_idx = min(idx + sample_batch_size, len(all_features))
        if idx % 50000 == 0:
            print(f"   Progress: {idx}/{len(all_features)} ({100*idx/len(all_features):.1f}%)")

        for sample_idx in range(idx, end_idx):
            # Normalize features
            for feat_idx in range(n_features):
                feat_min = feature_mins[feat_idx]
                feat_max = feature_maxs[feat_idx]

                invalid_mask = np.zeros(all_features[sample_idx].shape[0], dtype=bool)
                for marker in invalid_markers:
                    invalid_mask |= (all_features[sample_idx][:, feat_idx] == marker)

                if feat_max > feat_min:
                    all_features[sample_idx][:, feat_idx] = 2.0 * (all_features[sample_idx][:, feat_idx] - feat_min) / (feat_max - feat_min) - 1.0

                all_features[sample_idx][invalid_mask, feat_idx] = normalized_invalid_marker

            # Normalize target
            target_invalid = np.any(np.isin(all_targets[sample_idx], invalid_markers))
            if target_invalid:
                all_targets[sample_idx][:] = normalized_invalid_marker
            elif target_max > target_min:
                all_targets[sample_idx][:] = 2.0 * (all_targets[sample_idx] - target_min) / (target_max - target_min) - 1.0

    # Save final dataset
    print(f"\n8. Saving augmented dataset...")

    output_path = Path(data_dir) / "precomputed_sequences_augmented.npz"
    norm_stats_path = Path(data_dir) / "normalization_stats_augmented.npz"

    np.savez_compressed(
        output_path,
        features=all_features,
        targets=all_targets,
        masks=all_masks,
        target_stations=np.array(all_target_stations, dtype=np.int32),
        end_dates=np.array(all_end_dates, dtype=np.float64),
        start_dates=np.array(all_start_dates, dtype=np.float64),
        skip_pattern=np.array(all_skip_pattern, dtype=np.int32),
        permutation=np.array(all_permutation, dtype=np.int32),
        is_normalized=np.array([True], dtype=bool)
    )

    np.savez(
        norm_stats_path,
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
        target_min=target_min,
        target_max=target_max
    )

    # Cleanup
    print(f"\n9. Cleaning up temporary files...")
    shutil.rmtree(temp_dir)

    print(f"\n   Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1e9:.2f} GB")

    print("\n" + "=" * 70)
    print("✓ AUGMENTED DATASET COMPLETE!")
    print("=" * 70)
    print(f"Base samples: {len(base_dataset.sample_index):,}")
    print(f"Augmented samples: {len(all_features):,}")
    print(f"Augmentation factor: {len(all_features) / len(base_dataset.sample_index):.0f}x")
    print(f"\nPeak memory usage: ~{batch_size * total_augmentations * seq_length * total_features * 4 / 1e9:.1f} GB per batch")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_augmentations_batched(batch_size=100)
