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
import multiprocessing as mp
from functools import partial
from tqdm import tqdm


def _process_batch_direct_write(args):
    """
    Worker function that writes DIRECTLY to memmap files (no queue, no pickling!).

    Each worker receives paths to memmap files and writes to its assigned slice.
    No serialization of large arrays - just writes directly to disk.

    Can optionally normalize data using provided stats before writing.

    Args:
        args: Tuple of (batch_num, start_idx, batch_samples_data, aug_params, memmap_paths)

    Returns:
        batch_num (for tracking completion)
    """
    batch_num, start_idx, batch_samples_data, aug_params, memmap_paths = args

    # Unpack parameters
    seq_length, n_nearby_available, n_nearby_in_features = aug_params['dimensions']
    skip_patterns = aug_params['skip_patterns']
    all_permutations = aug_params['permutations']
    total_augmentations = aug_params['total_augmentations']
    target_features = aug_params['target_features']
    nearby_features_per_station = aug_params['nearby_features_per_station']
    total_features = aug_params['total_features']
    total_samples = aug_params['total_samples']

    # Normalization parameters (if provided)
    should_normalize = aug_params.get('normalize', False)
    if should_normalize:
        feature_mins = aug_params['feature_mins']
        feature_maxs = aug_params['feature_maxs']
        target_min = aug_params['target_min']
        target_max = aug_params['target_max']
        invalid_markers = aug_params.get('invalid_markers', [-9999.0, -1000.0])
        normalized_invalid_marker = -2.0

    batch_actual_size = len(batch_samples_data)
    batch_aug_size = batch_actual_size * total_augmentations
    end_idx = start_idx + batch_aug_size

    # Open memmap files in read-write mode (workers write to their assigned slices)
    all_features = np.lib.format.open_memmap(
        memmap_paths['features'], mode='r+',
        shape=(total_samples, seq_length, total_features)
    )
    all_targets = np.lib.format.open_memmap(
        memmap_paths['targets'], mode='r+',
        shape=(total_samples, 1)
    )
    all_masks = np.lib.format.open_memmap(
        memmap_paths['masks'], mode='r+',
        shape=(total_samples, seq_length, total_features)
    )
    all_target_stations = np.lib.format.open_memmap(
        memmap_paths['target_stations'], mode='r+',
        shape=(total_samples,)
    )
    all_skip_pattern = np.lib.format.open_memmap(
        memmap_paths['skip_pattern'], mode='r+',
        shape=(total_samples,)
    )
    all_permutation = np.lib.format.open_memmap(
        memmap_paths['permutation'], mode='r+',
        shape=(total_samples,)
    )

    # Load end/start dates (opened as regular arrays in worker, will be aggregated later)
    # For now, we'll return these to be collected by main process
    batch_end_dates = []
    batch_start_dates = []

    # DEBUG: Track what we're writing
    debug_mode = aug_params.get('debug', False)
    if debug_mode and batch_num == 0:
        print(f"\n[Worker Debug] Batch {batch_num}:")
        print(f"  start_idx={start_idx}, batch_samples={len(batch_samples_data)}")
        print(f"  total_augmentations={total_augmentations}")
        print(f"  expected end_idx={end_idx}")

    # Process augmentations
    current_idx = start_idx
    for sample_i, sample_data in enumerate(batch_samples_data):
        # Unpack pre-fetched sample
        base_features = sample_data['features']
        base_mask = sample_data['mask']
        base_target = sample_data['target']
        sample_info = sample_data['sample_info']

        if debug_mode and batch_num == 0 and sample_i == 0:
            print(f"  Sample 0 base_target: {base_target} (type: {type(base_target)}, shape: {getattr(base_target, 'shape', 'N/A')}, dtype: {getattr(base_target, 'dtype', 'N/A')})")
            print(f"  Sample 0 base_features shape: {base_features.shape}")
            print(f"  Sample 0 base_mask shape: {base_mask.shape}")

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

                # Normalize if stats provided (saves 2 hours!)
                normalized_target = base_target  # Use temp variable to avoid modifying base_target!
                if should_normalize:
                    # Normalize features (vectorized across timesteps)
                    for feat_idx in range(total_features):
                        feat_min = feature_mins[feat_idx]
                        feat_max = feature_maxs[feat_idx]
                        feat_data = aug_features[:, feat_idx]

                        # Find invalid markers
                        invalid_mask = np.zeros(len(feat_data), dtype=bool)
                        for marker in invalid_markers:
                            invalid_mask |= (feat_data == marker)

                        # Normalize valid values
                        if feat_max > feat_min:
                            aug_features[:, feat_idx] = 2.0 * (feat_data - feat_min) / (feat_max - feat_min) - 1.0

                        # Set invalid values
                        aug_features[invalid_mask, feat_idx] = normalized_invalid_marker

                    # Normalize target (use temp variable!)
                    if base_target not in invalid_markers:
                        if target_max > target_min:
                            normalized_target = 2.0 * (base_target - target_min) / (target_max - target_min) - 1.0
                    else:
                        normalized_target = normalized_invalid_marker

                # Write DIRECTLY to memmap (already normalized if requested!)
                all_features[current_idx] = aug_features
                all_targets[current_idx] = normalized_target
                all_masks[current_idx] = aug_mask
                all_target_stations[current_idx] = sample_info['target_station']
                all_skip_pattern[current_idx] = skip_idx
                all_permutation[current_idx] = perm_idx

                # DEBUG: Check first few writes
                if debug_mode and batch_num == 0 and sample_i == 0 and perm_idx < 2:
                    print(f"    Aug {skip_idx},{perm_idx}: base_target={base_target}, normalized_target={normalized_target}, writing to idx={current_idx}")
                    print(f"      Verification: all_targets[{current_idx}] = {all_targets[current_idx]}")

                # Collect dates (will save at the end)
                batch_end_dates.append(sample_info['end_date'].timestamp())
                batch_start_dates.append(sample_info['start_date'].timestamp())

                current_idx += 1

    # Flush this worker's writes to disk
    all_features.flush()
    all_targets.flush()
    all_masks.flush()
    all_target_stations.flush()
    all_skip_pattern.flush()
    all_permutation.flush()

    # Return batch info and dates (dates are small, safe to pickle)
    return (batch_num, start_idx, np.array(batch_end_dates, dtype=np.float64),
            np.array(batch_start_dates, dtype=np.float64))


# Remove old queue-based functions
# def _init_worker_with_queue(queue):
# def _writer_process(write_queue, ...):
# def _process_batch_to_queue(args):


def _process_samples_worker_v2(args):
    """
    Worker function that receives PRE-FETCHED samples (no dataset loading!)

    Args:
        args: Tuple of (batch_num, batch_samples_data, aug_params, batch_dir)
            batch_samples_data: List of dicts with 'features', 'mask', 'target', 'sample_info'

    Returns:
        Tuple of (batch_file_path, batch_size)
    """
    batch_num, batch_samples_data, aug_params, batch_dir = args

    # Unpack parameters
    seq_length, n_nearby_available, n_nearby_in_features = aug_params['dimensions']
    skip_patterns = aug_params['skip_patterns']
    all_permutations = aug_params['permutations']
    total_augmentations = aug_params['total_augmentations']
    target_features = aug_params['target_features']
    nearby_features_per_station = aug_params['nearby_features_per_station']
    total_features = aug_params['total_features']

    batch_actual_size = len(batch_samples_data)
    batch_aug_size = batch_actual_size * total_augmentations

    # Allocate arrays for this batch
    batch_features = np.zeros((batch_aug_size, seq_length, total_features), dtype=np.float32)
    batch_targets = np.zeros((batch_aug_size, 1), dtype=np.float32)
    batch_masks = np.zeros((batch_aug_size, seq_length, total_features), dtype=bool)
    batch_target_stations = []
    batch_end_dates = []
    batch_start_dates = []
    batch_skip_pattern = []
    batch_permutation = []

    aug_idx = 0

    for sample_data in batch_samples_data:
        # Unpack pre-fetched sample
        base_features = sample_data['features']
        base_mask = sample_data['mask']
        base_target = sample_data['target']
        sample_info = sample_data['sample_info']

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

                batch_target_stations.append(sample_info['target_station'])
                batch_end_dates.append(sample_info['end_date'].timestamp())
                batch_start_dates.append(sample_info['start_date'].timestamp())
                batch_skip_pattern.append(skip_idx)
                batch_permutation.append(perm_idx)

                aug_idx += 1

    # Save batch to file
    batch_file = Path(batch_dir) / f"batch_{batch_num:04d}.npz"
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

    return (batch_file, batch_aug_size)


def generate_all_augmentations_sequential(
    data_dir: str = "./meteogalicia_data",
    n_nearby_available: int = 5,
    n_nearby_in_features: int = 4,
    coverage_threshold: float = 0.25,
    seq_length: int = 64
):
    """
    Pre-compute ALL augmented samples SEQUENTIALLY (minimal memory!)

    No multiprocessing, no temp batches - writes directly to memory-mapped arrays.

    Memory usage: ~5GB (1 dataset + processing buffers)
    Speed: Slower than parallel, but no memory overhead

    Args:
        data_dir: Directory containing the MeteoGalicia dataset
        n_nearby_available: Number of nearby stations in base dataset (5)
        n_nearby_in_features: Number of nearby stations in augmented samples (4)
        coverage_threshold: Minimum coverage to include a parameter (0.25 = 25%)
        seq_length: Sequence length (default 64)
    """
    print("=" * 70)
    print("PRE-COMPUTING AUGMENTED DATASET (SEQUENTIAL - LOW MEMORY)")
    print("=" * 70)

    collector = MeteoGaliciaCollector(data_dir=data_dir)
    output_path = Path(data_dir) / "precomputed_sequences_augmented"
    norm_stats_path = Path(data_dir) / "normalization_stats_augmented.npz"

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
        seq_length=seq_length,
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

    # Calculate dimensions
    target_features = len(filtered_params)
    nearby_features_per_station = 1 + len(filtered_params) + 1
    total_features = target_features + (nearby_features_per_station * n_nearby_in_features)
    total_samples = len(base_dataset.sample_index) * total_augmentations

    print(f"   Total augmented samples: {total_samples:,}")
    print(f"   Sample shape: [{seq_length}, {total_features}]")

    # Calculate memory requirements
    features_size_gb = total_samples * seq_length * total_features * 4 / 1e9
    masks_size_gb = total_samples * seq_length * total_features * 1 / 1e9
    print(f"   Dataset size: {features_size_gb:.1f}GB features + {masks_size_gb:.1f}GB masks = {features_size_gb + masks_size_gb:.1f}GB total")

    # Create memory-mapped arrays (write directly, no temp batches!)
    print(f"\n4. Creating memory-mapped arrays...")
    output_path.mkdir(parents=True, exist_ok=True)

    # Use np.lib.format.open_memmap() to create proper .npy files with headers
    # (unlike np.memmap which creates raw binary files)
    all_features = np.lib.format.open_memmap(
        str(output_path / "features.npy"), dtype=np.float32, mode='w+',
        shape=(total_samples, seq_length, total_features)
    )
    all_targets = np.lib.format.open_memmap(
        str(output_path / "targets.npy"), dtype=np.float32, mode='w+',
        shape=(total_samples, 1)
    )
    all_masks = np.lib.format.open_memmap(
        str(output_path / "masks.npy"), dtype=bool, mode='w+',
        shape=(total_samples, seq_length, total_features)
    )
    all_target_stations = np.lib.format.open_memmap(
        str(output_path / "target_stations.npy"), dtype=np.int32, mode='w+',
        shape=(total_samples,)
    )
    all_skip_pattern = np.lib.format.open_memmap(
        str(output_path / "skip_pattern.npy"), dtype=np.int32, mode='w+',
        shape=(total_samples,)
    )
    all_permutation = np.lib.format.open_memmap(
        str(output_path / "permutation.npy"), dtype=np.int32, mode='w+',
        shape=(total_samples,)
    )

    # Use regular arrays for dates (small enough, will save with np.save later)
    all_end_dates = np.zeros(total_samples, dtype=np.float64)
    all_start_dates = np.zeros(total_samples, dtype=np.float64)

    # Process samples sequentially
    print(f"\n5. Generating augmentations (sequential)...")
    current_idx = 0

    for base_idx in tqdm(range(len(base_dataset.sample_index)), desc="   Processing", unit="sample"):
        # Get base sample
        sample = base_dataset[base_idx]
        base_features = sample['features'].numpy()
        base_mask = sample['mask'].numpy()
        base_target = sample['target'].numpy()
        sample_info = base_dataset.sample_index[base_idx]

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

                # Write directly to memmap
                all_features[current_idx] = aug_features
                all_targets[current_idx] = base_target
                all_masks[current_idx] = aug_mask
                all_target_stations[current_idx] = sample_info['target_station']
                all_end_dates[current_idx] = sample_info['end_date'].timestamp()
                all_start_dates[current_idx] = sample_info['start_date'].timestamp()
                all_skip_pattern[current_idx] = skip_idx
                all_permutation[current_idx] = perm_idx

                current_idx += 1

        # Flush periodically to avoid buffer buildup
        if (base_idx + 1) % 1000 == 0:
            all_features.flush()
            all_targets.flush()
            all_masks.flush()

    print(f"   ✓ Generated {current_idx:,} augmented samples")

    # Compute normalization statistics
    print(f"\n6. Computing normalization statistics...")
    feature_mins = np.full(total_features, np.inf, dtype=np.float32)
    feature_maxs = np.full(total_features, -np.inf, dtype=np.float32)
    invalid_markers = [-9999.0, -1000.0]

    sample_batch_size = 10000
    for i in tqdm(range(0, len(all_features), sample_batch_size),
                  desc="   Computing stats",
                  unit="batch",
                  total=(len(all_features) + sample_batch_size - 1) // sample_batch_size):
        end_i = min(i + sample_batch_size, len(all_features))
        features_batch = all_features[i:end_i]
        masks_batch = all_masks[i:end_i]

        for feat_idx in range(total_features):
            feat_data = features_batch[:, :, feat_idx]
            feat_mask = masks_batch[:, :, feat_idx]

            valid_mask = feat_mask
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

    # Normalize
    print(f"\n7. Normalizing augmented samples (vectorized)...")
    normalized_invalid_marker = -2.0

    for idx in tqdm(range(0, len(all_features), sample_batch_size),
                    desc="   Normalizing",
                    unit="batch",
                    total=(len(all_features) + sample_batch_size - 1) // sample_batch_size):
        end_idx = min(idx + sample_batch_size, len(all_features))

        features_batch = all_features[idx:end_idx]
        targets_batch = all_targets[idx:end_idx]

        for feat_idx in range(total_features):
            feat_min = feature_mins[feat_idx]
            feat_max = feature_maxs[feat_idx]

            feat_data = features_batch[:, :, feat_idx]

            invalid_mask = np.zeros_like(feat_data, dtype=bool)
            for marker in invalid_markers:
                invalid_mask |= (feat_data == marker)

            if feat_max > feat_min:
                features_batch[:, :, feat_idx] = 2.0 * (feat_data - feat_min) / (feat_max - feat_min) - 1.0

            features_batch[invalid_mask, feat_idx] = normalized_invalid_marker

        target_invalid_mask = np.zeros(len(targets_batch), dtype=bool)
        for marker in invalid_markers:
            target_invalid_mask |= (targets_batch == marker).flatten()

        if target_max > target_min:
            targets_batch[:] = 2.0 * (targets_batch - target_min) / (target_max - target_min) - 1.0

        targets_batch[target_invalid_mask] = normalized_invalid_marker

        all_features[idx:end_idx] = features_batch
        all_targets[idx:end_idx] = targets_batch

        if (end_idx % 100000) < sample_batch_size:
            all_features.flush()
            all_targets.flush()

    # Flush and save
    print(f"\n8. Saving dataset...")
    all_features.flush()
    all_targets.flush()
    all_masks.flush()
    all_target_stations.flush()
    all_skip_pattern.flush()
    all_permutation.flush()

    # Save dates with np.save() (matches non-augmented dataset format)
    np.save(output_path / 'end_dates.npy', all_end_dates)
    np.save(output_path / 'start_dates.npy', all_start_dates)
    np.save(output_path / 'is_normalized.npy', np.array([True], dtype=bool))

    np.savez(
        norm_stats_path,
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
        target_min=target_min,
        target_max=target_max
    )

    print(f"\n   ✓ Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("✓ AUGMENTED DATASET COMPLETE!")
    print("=" * 70)
    print(f"Base samples: {len(base_dataset.sample_index):,}")
    print(f"Augmented samples: {total_samples:,}")
    print(f"Augmentation factor: {total_samples / len(base_dataset.sample_index):.0f}x")
    print(f"\nMemory usage:")
    print(f"  - Peak RAM: ~5 GB (sequential processing)")
    print(f"  - Disk space: {(features_size_gb + masks_size_gb):.1f} GB")
    print("=" * 70)


def generate_all_augmentations_batched(
    data_dir: str = "./meteogalicia_data",
    n_nearby_available: int = 5,
    n_nearby_in_features: int = 4,
    coverage_threshold: float = 0.25,
    seq_length: int = 64,
    batch_size: int = 1000,
    num_workers: int = None,  # Auto-detect: physical cores (avoids hyperthreading)
    use_base_stats: bool = False  # Use base dataset stats (saves ~2 hours!)
):
    """
    Pre-compute ALL augmented samples with batched processing (memory efficient!)

    Uses multiprocessing with PRE-FETCHED SAMPLES (no dataset in workers!):
    - Main process loads dataset ONCE and fetches samples
    - Workers receive pre-fetched samples (no dataset loading)
    - Memory usage: ~4GB (main dataset) + ~500MB per worker

    Args:
        data_dir: Directory containing the MeteoGalicia dataset
        n_nearby_available: Number of nearby stations in base dataset (5)
        n_nearby_in_features: Number of nearby stations in augmented samples (4)
        coverage_threshold: Minimum coverage to include a parameter (0.25 = 25%)
        seq_length: Sequence length (default 64)
        batch_size: Number of base samples to process per batch
        num_workers: Number of parallel workers (None = auto-detect physical cores)
        use_base_stats: If True, use base dataset normalization stats and normalize
                        in workers during generation (saves ~2 hours!)
    """
    print("=" * 70)
    print("PRE-COMPUTING AUGMENTED DATASET (MEMORY EFFICIENT)")
    print("=" * 70)

    collector = MeteoGaliciaCollector(data_dir=data_dir)
    batch_dir = Path(data_dir) / "augmented_batches"
    output_path = Path(data_dir) / "precomputed_sequences_augmented"
    norm_stats_path = Path(data_dir) / "normalization_stats_augmented.npz"

    # Get filtered parameters
    print("\n1. Analyzing parameter coverage...")
    coverage, filtered_params = collector.analyze_parameter_coverage(
        coverage_threshold=coverage_threshold
    )

    print(f"   Selected {len(filtered_params)} parameters")

    # Load base dataset (in main process only!)
    print(f"\n2. Loading base dataset with {n_nearby_available} nearby stations...")
    dense_path = Path(data_dir) / "dense_features.npz"

    base_dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=seq_length,
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

    # Create batch info tuples for all batches
    # Note: No longer passing dataset_params in batch_info since dataset is loaded in initializer
    target_features = len(filtered_params)
    nearby_features_per_station = 1 + len(filtered_params) + 1
    total_features = target_features + (nearby_features_per_station * n_nearby_in_features)
    aug_params = {
        'dimensions': (seq_length, n_nearby_available, n_nearby_in_features),
        'skip_patterns': skip_patterns,
        'permutations': all_permutations,
        'filtered_params': filtered_params,
        'total_augmentations': total_augmentations,
        'target_features': target_features,
        'nearby_features_per_station': nearby_features_per_station,
        'total_features': total_features,
        'debug': True  # ENABLED for debugging data corruption
    }
    # Calculate number of batches from dataset size (reliable!)
    num_batches = (len(base_dataset.sample_index) + batch_size - 1) // batch_size
    total_samples = len(base_dataset.sample_index) * total_augmentations

    print(f"   Total augmented samples: {total_samples:,}")
    print(f"   Sample shape: [{seq_length}, {total_features}]")

    # Auto-detect number of workers (avoid hyperthreading)
    if num_workers is None:
        # mp.cpu_count() returns logical cores (includes hyperthreading)
        # Divide by 2 to get physical cores, then subtract 1 for breathing room
        logical_cores = mp.cpu_count()
        num_workers = max(1, (logical_cores // 2) - 1)
        print(f"   Auto-detected {num_workers} workers (physical cores - 1)")

    # Compute augmented stats from base dataset if requested (saves ~2 hours!)
    base_stats = None
    if use_base_stats:
        print(f"\n4. Computing augmented dataset stats from base dataset (5 nearby)...")
        print(f"   Key insight: Since augmentation permutes 4 of 5 nearby stations,")
        print(f"   each nearby slot can be ANY of the 5 stations.")
        print(f"   Therefore: Range for each nearby feature = min/max across ALL 5 stations.")
        print()

        # Sample from base dataset to compute stats
        num_samples_for_stats = min(10000, len(base_dataset.sample_index))
        sample_indices = np.random.choice(len(base_dataset.sample_index),
                                         size=num_samples_for_stats, replace=False)

        print(f"   Sampling {num_samples_for_stats} samples from base dataset...")

        # Initialize min/max tracking
        target_features_count = len(filtered_params)
        nearby_features_per_station = 1 + len(filtered_params) + 1  # distance + features + soil
        augmented_total_features = target_features_count + (nearby_features_per_station * n_nearby_in_features)

        feature_mins = np.full(augmented_total_features, np.inf, dtype=np.float32)
        feature_maxs = np.full(augmented_total_features, -np.inf, dtype=np.float32)
        target_min = np.inf
        target_max = -np.inf

        for idx in tqdm(sample_indices, desc="   Computing stats"):
            sample = base_dataset[int(idx)]
            features = sample['features'].numpy()
            target = sample['target'].numpy()[0]

            # Target stats
            if target != -1000.0 and target != -9999.0:
                target_min = min(target_min, target)
                target_max = max(target_max, target)

            # Target station features (unchanged)
            target_feats = features[:, :target_features_count]
            for feat_idx in range(target_features_count):
                feat_values = target_feats[:, feat_idx]
                valid = feat_values[(feat_values != -1000.0) & (feat_values != -9999.0)]
                if len(valid) > 0:
                    feature_mins[feat_idx] = min(feature_mins[feat_idx], valid.min())
                    feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid.max())

            # Nearby stations: Extract all 5 stations' data
            nearby_start = target_features_count
            nearby_base = features[:, nearby_start:].reshape(seq_length, n_nearby_available,
                                                            nearby_features_per_station)

            # For each feature across nearby stations (distance, features, soil):
            # The augmented dataset will have 4 stations, each slot can be ANY of the 5
            for nearby_feat_idx in range(nearby_features_per_station):
                feat_across_stations = nearby_base[:, :, nearby_feat_idx]
                valid = feat_across_stations[(feat_across_stations != -1000.0) &
                                            (feat_across_stations != -9999.0)]

                if len(valid) > 0:
                    # Apply same range to all 4 slots
                    for slot in range(n_nearby_in_features):
                        aug_feat_idx = target_features_count + (slot * nearby_features_per_station) + nearby_feat_idx
                        feature_mins[aug_feat_idx] = min(feature_mins[aug_feat_idx], valid.min())
                        feature_maxs[aug_feat_idx] = max(feature_maxs[aug_feat_idx], valid.max())

        print(f"   ✓ Computed augmented stats from base dataset")
        print(f"   Feature range: [{feature_mins[~np.isinf(feature_mins)].min():.2f}, {feature_maxs[~np.isinf(feature_maxs)].max():.2f}]")
        print(f"   Target range: [{target_min:.2f}, {target_max:.2f}]")
        print(f"   → Workers will normalize data during generation (saves ~2 hours!)")

        # Create stats dict
        base_stats = {
            'feature_mins': feature_mins,
            'feature_maxs': feature_maxs,
            'target_min': target_min,
            'target_max': target_max
        }

        # Add stats to aug_params for workers
        aug_params['normalize'] = True
        aug_params['feature_mins'] = feature_mins
        aug_params['feature_maxs'] = feature_maxs
        aug_params['target_min'] = float(target_min)
        aug_params['target_max'] = float(target_max)
        aug_params['invalid_markers'] = [-9999.0, -1000.0]

    print(f"\n{'5' if use_base_stats else '4'}. Creating memmap files and processing in parallel (direct write, no serialization!)...")
    print(f"   Total batches: {num_batches}")
    print(f"   Estimated memory: ~4GB (dataset) + ~{num_workers * 0.5:.1f}GB (workers) = ~{4 + num_workers * 0.5:.1f}GB total")
    print(f"   Estimated speedup: ~{num_workers}x faster than sequential")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Create memmap files FIRST (before workers start)
    print(f"   Creating memmap files...")
    memmap_paths = {
        'features': str(output_path / "features.npy"),
        'targets': str(output_path / "targets.npy"),
        'masks': str(output_path / "masks.npy"),
        'target_stations': str(output_path / "target_stations.npy"),
        'skip_pattern': str(output_path / "skip_pattern.npy"),
        'permutation': str(output_path / "permutation.npy")
    }

    # Create all memmap files with 'w+' mode
    np.lib.format.open_memmap(
        memmap_paths['features'], dtype=np.float32, mode='w+',
        shape=(total_samples, seq_length, total_features)
    )
    np.lib.format.open_memmap(
        memmap_paths['targets'], dtype=np.float32, mode='w+',
        shape=(total_samples, 1)
    )
    np.lib.format.open_memmap(
        memmap_paths['masks'], dtype=bool, mode='w+',
        shape=(total_samples, seq_length, total_features)
    )
    np.lib.format.open_memmap(
        memmap_paths['target_stations'], dtype=np.int32, mode='w+',
        shape=(total_samples,)
    )
    np.lib.format.open_memmap(
        memmap_paths['skip_pattern'], dtype=np.int32, mode='w+',
        shape=(total_samples,)
    )
    np.lib.format.open_memmap(
        memmap_paths['permutation'], dtype=np.int32, mode='w+',
        shape=(total_samples,)
    )

    # Add total_samples to aug_params for workers
    aug_params['total_samples'] = total_samples

    # Process batches in parallel with lazy sample fetching
    print(f"   Starting parallel processing (workers write directly to memmap)...")

    def batch_generator():
        """Generator that fetches samples on-demand and tracks augmented sample positions"""
        current_aug_idx = 0  # Track position in augmented dataset
        for batch_num in range(num_batches):
            start_idx_base = batch_num * batch_size
            end_idx_base = min(start_idx_base + batch_size, len(base_dataset.sample_index))

            # Fetch samples for this batch
            batch_samples_data = []
            for idx in range(start_idx_base, end_idx_base):
                sample = base_dataset[idx]
                sample_info = base_dataset.sample_index[idx]

                batch_samples_data.append({
                    'features': sample['features'].numpy(),
                    'mask': sample['mask'].numpy(),
                    'target': sample['target'].numpy(),
                    'sample_info': sample_info
                })

            # Calculate where this batch will be written in augmented dataset
            start_idx_aug = current_aug_idx
            current_aug_idx += len(batch_samples_data) * total_augmentations

            yield (batch_num, start_idx_aug, batch_samples_data, aug_params, memmap_paths)

    # Process batches in parallel (workers write DIRECTLY to memmap, no pickling!)
    all_end_dates = np.zeros(total_samples, dtype=np.float64)
    all_start_dates = np.zeros(total_samples, dtype=np.float64)

    with mp.Pool(processes=num_workers) as pool:
        for batch_num, start_idx, end_dates, start_dates in tqdm(
            pool.imap(_process_batch_direct_write, batch_generator(), chunksize=1),
            total=num_batches,
            desc="      Processing batches",
            unit="batch"
        ):
            # Collect dates from workers
            end_idx = start_idx + len(end_dates)
            all_end_dates[start_idx:end_idx] = end_dates
            all_start_dates[start_idx:end_idx] = start_dates

    # Save date arrays
    np.save(output_path / 'end_dates.npy', all_end_dates)
    np.save(output_path / 'start_dates.npy', all_start_dates)

    print(f"   ✓ All {num_batches} batches processed and written!")

    # Calculate memory requirements (bool masks = 1 byte, float32 = 4 bytes)
    features_size_gb = total_samples * seq_length * total_features * 4 / 1e9
    masks_size_gb = total_samples * seq_length * total_features * 1 / 1e9  # bool = 1 byte!
    step_num = 6 if use_base_stats else 5
    print(f"\n{step_num}. Data written to memmap files:")
    print(f"   Dataset size: {features_size_gb:.1f}GB features + {masks_size_gb:.1f}GB masks = {features_size_gb + masks_size_gb:.1f}GB total")
    print(f"   Masks using bool dtype (75% smaller than float32!)")

    # If using base stats, skip statistics computation and normalization (already done in workers!)
    if use_base_stats:
        print(f"\n{step_num + 1}. Skipping statistics computation (using base dataset stats)")
        print(f"   ✓ Data already normalized by workers")

        # Save normalization flag
        np.save(output_path / "is_normalized.npy", np.array([True], dtype=bool))

        # Save normalization stats (copy from base dataset)
        np.savez(
            norm_stats_path,
            feature_mins=base_stats['feature_mins'],
            feature_maxs=base_stats['feature_maxs'],
            target_min=base_stats['target_min'],
            target_max=base_stats['target_max']
        )
        print(f"   ✓ Saved normalization stats (copied from base dataset)")
        print(f"   ✓ Saved to: {output_path}")

        print("\n" + "=" * 70)
        print("✓ AUGMENTED DATASET COMPLETE!")
        print("=" * 70)
        print(f"Base samples: {len(base_dataset.sample_index):,}")
        print(f"Augmented samples: {total_samples:,}")
        print(f"Augmentation factor: {total_samples / len(base_dataset.sample_index):.0f}x")
        print(f"\nMemory usage:")
        print(f"  - Dataset (main process): ~4 GB")
        print(f"  - Workers: ~{num_workers * 0.5:.1f} GB")
        print(f"  - Total peak RAM: ~{4 + num_workers * 0.5:.1f} GB")
        print(f"  - Disk space: ~{(features_size_gb + masks_size_gb):.1f} GB")
        print(f"\nPerformance:")
        print(f"  - Normalization: Done in workers (SAVED ~2 hours!)")
        print("=" * 70)
        return

    # Load memmap files for statistics and normalization
    print(f"\n{step_num + 1}. Loading memmap files for statistics...")
    all_features = np.lib.format.open_memmap(
        str(output_path / "features.npy"), mode='r+'  # Read-write for normalization
    )
    all_targets = np.lib.format.open_memmap(
        str(output_path / "targets.npy"), mode='r+'
    )
    all_masks = np.lib.format.open_memmap(
        str(output_path / "masks.npy"), mode='r'  # Read-only
    )

    # Try to load base dataset normalization stats for comparison
    print(f"\n{step_num + 2}. Comparing with base dataset normalization stats...")
    base_norm_stats_path = Path(data_dir) / "normalization_stats.npz"

    if base_norm_stats_path.exists():
        print(f"   Found base dataset stats, comparing...")
        base_stats_for_comparison = np.load(base_norm_stats_path)
        base_feature_mins = base_stats_for_comparison['feature_mins']
        base_feature_maxs = base_stats_for_comparison['feature_maxs']
        base_target_min = float(base_stats_for_comparison['target_min'])
        base_target_max = float(base_stats_for_comparison['target_max'])
    else:
        print(f"   Base dataset stats not found, will compute from scratch")
        base_feature_mins = None
        base_feature_maxs = None

    # Compute augmented dataset statistics
    print(f"\n{step_num + 3}. Computing augmented dataset normalization statistics...")

    n_features = all_features.shape[2]
    feature_mins = np.full(n_features, np.inf, dtype=np.float32)
    feature_maxs = np.full(n_features, -np.inf, dtype=np.float32)
    invalid_markers = [-9999.0, -1000.0]

    # Sample-based statistics computation (memory efficient)
    sample_batch_size = 10000
    for i in tqdm(range(0, len(all_features), sample_batch_size),
                  desc="   Computing stats",
                  unit="batch",
                  total=(len(all_features) + sample_batch_size - 1) // sample_batch_size):
        end_i = min(i + sample_batch_size, len(all_features))
        features_batch = all_features[i:end_i]
        masks_batch = all_masks[i:end_i]

        for feat_idx in range(n_features):
            feat_data = features_batch[:, :, feat_idx]
            feat_mask = masks_batch[:, :, feat_idx]

            # Create a copy for modification (masks are read-only)
            valid_mask = feat_mask.copy()
            for marker in invalid_markers:
                valid_mask &= (feat_data != marker)

            valid_data = feat_data[valid_mask]

            if len(valid_data) > 0:
                feature_mins[feat_idx] = min(feature_mins[feat_idx], valid_data.min())
                feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid_data.max())

    valid_targets = all_targets[~np.isin(all_targets, invalid_markers)]
    target_min = valid_targets.min() if len(valid_targets) > 0 else 0.0
    target_max = valid_targets.max() if len(valid_targets) > 0 else 1.0

    print(f"   Augmented feature range: [{feature_mins.min():.2f}, {feature_maxs.max():.2f}]")
    print(f"   Augmented target range: [{target_min:.2f}, {target_max:.2f}]")

    # Compare with base dataset stats
    if base_feature_mins is not None:
        print(f"\n   Comparing augmented vs base dataset stats:")

        # Calculate differences
        min_diffs = np.abs(feature_mins - base_feature_mins)
        max_diffs = np.abs(feature_maxs - base_feature_maxs)
        target_min_diff = abs(target_min - base_target_min)
        target_max_diff = abs(target_max - base_target_max)

        # Calculate relative differences (as percentage of range)
        feature_ranges = feature_maxs - feature_mins
        rel_min_diffs = np.where(feature_ranges > 0, min_diffs / feature_ranges * 100, 0)
        rel_max_diffs = np.where(feature_ranges > 0, max_diffs / feature_ranges * 100, 0)

        print(f"   Base feature range: [{base_feature_mins.min():.2f}, {base_feature_maxs.max():.2f}]")
        print(f"   Base target range: [{base_target_min:.2f}, {base_target_max:.2f}]")
        print(f"   Feature min diff: max={min_diffs.max():.6f}, mean={min_diffs.mean():.6f}")
        print(f"   Feature max diff: max={max_diffs.max():.6f}, mean={max_diffs.mean():.6f}")
        print(f"   Relative diff (%): max_min={rel_min_diffs.max():.3f}%, max_max={rel_max_diffs.max():.3f}%")
        print(f"   Target min diff: {target_min_diff:.6f}")
        print(f"   Target max diff: {target_max_diff:.6f}")

        # Check if differences are negligible (< 0.1% of range)
        if rel_min_diffs.max() < 0.1 and rel_max_diffs.max() < 0.1:
            print(f"   ✓ Stats are nearly identical! (<0.1% difference)")
            print(f"   → Could use base stats directly in future runs (would save ~1 hour)")
        elif rel_min_diffs.max() < 1.0 and rel_max_diffs.max() < 1.0:
            print(f"   ✓ Stats are very close (<1% difference)")
            print(f"   → Could potentially use base stats with minor impact")
        else:
            print(f"   ⚠ Stats differ significantly (>1% difference)")
            print(f"   → Should compute augmented stats (current approach)")

    # Also compute the "correct" augmented stats from base dataset for comparison
    print(f"\n   Computing correct augmented stats from base dataset (5 nearby) for validation...")
    num_samples_validation = min(1000, len(base_dataset.sample_index))
    validation_indices = np.random.choice(len(base_dataset.sample_index),
                                         size=num_samples_validation, replace=False)

    target_features_count = len(filtered_params)
    nearby_features_per_station = 1 + len(filtered_params) + 1
    expected_total_features = target_features_count + (nearby_features_per_station * n_nearby_in_features)

    expected_feature_mins = np.full(expected_total_features, np.inf, dtype=np.float32)
    expected_feature_maxs = np.full(expected_total_features, -np.inf, dtype=np.float32)
    expected_target_min = np.inf
    expected_target_max = -np.inf

    for idx in validation_indices:
        sample = base_dataset[int(idx)]
        features = sample['features'].numpy()
        target = sample['target'].numpy()[0]

        # Target stats
        if target != -1000.0 and target != -9999.0:
            expected_target_min = min(expected_target_min, target)
            expected_target_max = max(expected_target_max, target)

        # Target station features
        target_feats = features[:, :target_features_count]
        for feat_idx in range(target_features_count):
            feat_values = target_feats[:, feat_idx]
            valid = feat_values[(feat_values != -1000.0) & (feat_values != -9999.0)]
            if len(valid) > 0:
                expected_feature_mins[feat_idx] = min(expected_feature_mins[feat_idx], valid.min())
                expected_feature_maxs[feat_idx] = max(expected_feature_maxs[feat_idx], valid.max())

        # Nearby stations: all 5 stations' data
        nearby_start = target_features_count
        nearby_base = features[:, nearby_start:].reshape(seq_length, n_nearby_available,
                                                        nearby_features_per_station)

        for nearby_feat_idx in range(nearby_features_per_station):
            feat_across_stations = nearby_base[:, :, nearby_feat_idx]
            valid = feat_across_stations[(feat_across_stations != -1000.0) &
                                        (feat_across_stations != -9999.0)]

            if len(valid) > 0:
                for slot in range(n_nearby_in_features):
                    aug_feat_idx = target_features_count + (slot * nearby_features_per_station) + nearby_feat_idx
                    expected_feature_mins[aug_feat_idx] = min(expected_feature_mins[aug_feat_idx], valid.min())
                    expected_feature_maxs[aug_feat_idx] = max(expected_feature_maxs[aug_feat_idx], valid.max())

    # Compare expected vs actual
    print(f"\n   Validation: Comparing expected (from base 5 nearby) vs actual augmented stats:")

    # Only compare valid indices (not inf)
    valid_mask = ~(np.isinf(expected_feature_mins) | np.isinf(expected_feature_maxs) |
                   np.isinf(feature_mins) | np.isinf(feature_maxs))

    if valid_mask.sum() > 0:
        expected_min_diffs = np.abs(expected_feature_mins[valid_mask] - feature_mins[valid_mask])
        expected_max_diffs = np.abs(expected_feature_maxs[valid_mask] - feature_maxs[valid_mask])

        print(f"   Expected feature range: [{expected_feature_mins[valid_mask].min():.2f}, {expected_feature_maxs[valid_mask].max():.2f}]")
        print(f"   Actual feature range: [{feature_mins[valid_mask].min():.2f}, {feature_maxs[valid_mask].max():.2f}]")
        print(f"   Feature min diff: max={expected_min_diffs.max():.6f}, mean={expected_min_diffs.mean():.6f}")
        print(f"   Feature max diff: max={expected_max_diffs.max():.6f}, mean={expected_max_diffs.mean():.6f}")

        if expected_min_diffs.max() < 1.0 and expected_max_diffs.max() < 1.0:
            print(f"   ✅ EXCELLENT! Actual stats match expected augmented stats (<1.0 absolute diff)")
            print(f"   → --use-base-stats with correct computation would work perfectly!")
        else:
            print(f"   ⚠️  Stats differ - may need more validation samples or check algorithm")

    # Normalize in batches (working with memory-mapped arrays)
    # VECTORIZED VERSION - much faster than sample-by-sample loops!
    print(f"\n{step_num + 4}. Normalizing augmented samples (vectorized)...")

    normalized_invalid_marker = -2.0

    for idx in tqdm(range(0, len(all_features), sample_batch_size),
                    desc="   Normalizing",
                    unit="batch",
                    total=(len(all_features) + sample_batch_size - 1) // sample_batch_size):
        end_idx = min(idx + sample_batch_size, len(all_features))

        # Load batch into memory for fast vectorized operations
        features_batch = all_features[idx:end_idx]  # Shape: [batch_size, seq_length, n_features]
        targets_batch = all_targets[idx:end_idx]    # Shape: [batch_size, 1]

        # Normalize features - VECTORIZED across all samples and timesteps for each feature
        for feat_idx in range(n_features):
            feat_min = feature_mins[feat_idx]
            feat_max = feature_maxs[feat_idx]

            # Get all data for this feature across batch
            feat_data = features_batch[:, :, feat_idx]  # Shape: [batch_size, seq_length]

            # Find invalid markers (vectorized!)
            invalid_mask = np.zeros_like(feat_data, dtype=bool)
            for marker in invalid_markers:
                invalid_mask |= (feat_data == marker)

            # Normalize all valid values at once (vectorized!)
            if feat_max > feat_min:
                features_batch[:, :, feat_idx] = 2.0 * (feat_data - feat_min) / (feat_max - feat_min) - 1.0

            # Set invalid values to marker (vectorized!)
            features_batch[invalid_mask, feat_idx] = normalized_invalid_marker

        # Normalize targets (vectorized across batch!)
        target_invalid_mask = np.zeros(len(targets_batch), dtype=bool)
        for marker in invalid_markers:
            target_invalid_mask |= (targets_batch == marker).flatten()

        # Normalize valid targets
        if target_max > target_min:
            targets_batch[:] = 2.0 * (targets_batch - target_min) / (target_max - target_min) - 1.0

        # Set invalid targets
        targets_batch[target_invalid_mask] = normalized_invalid_marker

        # Write normalized batch back to memmap (this is the only slow part - disk I/O)
        all_features[idx:end_idx] = features_batch
        all_targets[idx:end_idx] = targets_batch

        # Flush to disk periodically (less frequently since we're faster)
        if (end_idx % 100000) < sample_batch_size:
            all_features.flush()
            all_targets.flush()

    # Save final dataset
    print(f"\n{step_num + 5}. Flushing all changes to disk...")
    all_features.flush()
    all_targets.flush()

    # Save normalization flag
    np.save(output_path / "is_normalized.npy", np.array([True], dtype=bool))

    # Save normalization stats
    np.savez(
        norm_stats_path,
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
        target_min=target_min,
        target_max=target_max
    )

    print(f"\n   ✓ Saved to: {output_path}")

    print("\n" + "=" * 70)
    print("✓ AUGMENTED DATASET COMPLETE!")
    print("=" * 70)
    print(f"Base samples: {len(base_dataset.sample_index):,}")
    print(f"Augmented samples: {total_samples:,}")
    print(f"Augmentation factor: {total_samples / len(base_dataset.sample_index):.0f}x")
    print(f"\nMemory usage:")
    print(f"  - Dataset (main process): ~4 GB")
    print(f"  - Workers: ~{num_workers * 0.5:.1f} GB")
    print(f"  - Total peak RAM: ~{4 + num_workers * 0.5:.1f} GB")
    print(f"  - Disk space: ~{(features_size_gb + masks_size_gb):.1f} GB")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    # Parse command-line arguments
    use_sequential = False
    use_base_stats = False

    for arg in sys.argv[1:]:
        if arg == "--sequential":
            use_sequential = True
        elif arg == "--use-base-stats":
            use_base_stats = True

    # Choose mode based on arguments
    if use_sequential:
        # Sequential mode: ~5GB RAM (slow but minimal memory)
        print("Using SEQUENTIAL mode (minimal memory)")
        if use_base_stats:
            print("WARNING: --use-base-stats is not supported in sequential mode (ignored)")
        generate_all_augmentations_sequential()
    else:
        # Batched mode: auto-detects CPU cores (faster)
        print("Using BATCHED mode (parallel, auto-detect workers)")
        if use_base_stats:
            print("Using base dataset stats (saves ~2 hours!)")
        print("Tip: Use --sequential for systems with <8GB RAM")
        print("Tip: Use --use-base-stats to skip statistics computation and normalization")
        generate_all_augmentations_batched(batch_size=100, use_base_stats=use_base_stats)
