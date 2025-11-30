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
    num_workers: int = None  # Auto-detect: physical cores (avoids hyperthreading)
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
        'total_features': total_features
    }
    # Calculate number of batches from dataset size (reliable!)
    num_batches = (len(base_dataset.sample_index) + batch_size - 1) // batch_size

    # If resuming from existing batches, verify count matches
    if batch_dir.exists():
        existing_batch_files = list(batch_dir.glob('batch_*.npz'))  # Only count batch_XXXX.npz files
        if existing_batch_files:
            existing_count = len(existing_batch_files)
            if existing_count != num_batches:
                print(f"   ⚠ Warning: Found {existing_count} existing batches, but expected {num_batches}")
                print(f"   Using existing batches count: {existing_count}")
                num_batches = existing_count

    batch_infos = []
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(base_dataset.sample_index))
        batch_infos.append((batch_num, start_idx, end_idx, aug_params, str(batch_dir)))

    if not batch_dir.exists():
        print(f"   Total augmented samples: {len(base_dataset.sample_index) * total_augmentations:,}")
        # Get dimensions
        # sample0 = base_dataset[0]


        print(f"   Sample shape: [{seq_length}, {total_features}]")

        # Create temporary directory for batch files under data_dir (not /tmp)

        batch_dir.mkdir(exist_ok=True)

        # Auto-detect number of workers (avoid hyperthreading)
        if num_workers is None:
            # mp.cpu_count() returns logical cores (includes hyperthreading)
            # Divide by 2 to get physical cores, then subtract 1 for breathing room
            logical_cores = mp.cpu_count()
            num_workers = max(1, (logical_cores // 2) - 1)
            print(f"   Auto-detected {num_workers} workers (physical cores - 1)")

        print(f"\n4. Processing in batches of {batch_size} using {num_workers} CPU cores...")
        print(f"   Batch directory: {batch_dir}")
        print(f"   Total batches: {num_batches}")
        print(f"   Estimated memory: ~4GB (dataset) + ~{num_workers * 0.5:.1f}GB (workers) = ~{4 + num_workers * 0.5:.1f}GB total")
        print(f"   Estimated speedup: ~{num_workers}x faster than sequential")

        # Process batches in parallel with lazy sample fetching (overlaps fetch + process!)
        print(f"   Starting parallel processing (fetch overlapped with processing)...")

        def batch_generator():
            """Generator that fetches samples on-demand, overlapping with worker processing"""
            for batch_num in range(num_batches):
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, len(base_dataset.sample_index))

                # Fetch samples for this batch (done lazily as workers request work)
                batch_samples_data = []
                for idx in range(start_idx, end_idx):
                    sample = base_dataset[idx]
                    sample_info = base_dataset.sample_index[idx]

                    # Store as numpy arrays (will be pickled when sent to worker)
                    batch_samples_data.append({
                        'features': sample['features'].numpy(),
                        'mask': sample['mask'].numpy(),
                        'target': sample['target'].numpy(),
                        'sample_info': sample_info
                    })

                yield (batch_num, batch_samples_data, aug_params, str(batch_dir))

        # Process batches in parallel (fetching overlaps with processing!)
        with mp.Pool(processes=num_workers) as pool:
            batch_files = []
            batch_sizes = []
            for batch_file, batch_aug_size in tqdm(pool.imap(_process_samples_worker_v2, batch_generator(), chunksize=1),
                                                     total=num_batches,
                                                     desc="      Processing (fetch+augment)",
                                                     unit="batch"):
                batch_files.append(batch_file)
                batch_sizes.append(batch_aug_size)

        print(f"   ✓ All {num_batches} batches processed!")

        # Save batch sizes metadata to avoid re-loading files later
        batch_sizes_file = batch_dir / "_batch_sizes.npy"
        np.save(batch_sizes_file, np.array(batch_sizes, dtype=np.int32))

    # Merge all batches - TRULY MEMORY EFFICIENT VERSION using memory-mapped arrays
    print(f"\n5. Merging batches...")
    print(f"   Using memory-mapped arrays to avoid loading everything into RAM...")

    # Load batch files list if resuming
    if not batch_dir.exists():
        raise RuntimeError("Batch directory not found - cannot merge without batches!")

    batch_files = sorted(batch_dir.glob("batch_*.npz"))

    # Calculate total size from saved metadata (avoids re-loading files!)
    print(f"   Pass 1/2: Calculating total size from metadata...")
    batch_sizes_file = batch_dir / "_batch_sizes.npy"

    if batch_sizes_file.exists():
        # Use saved batch sizes (FAST!)
        batch_sizes = np.load(batch_sizes_file)
        total_samples = int(batch_sizes.sum())
        print(f"   ✓ Loaded batch sizes from metadata (avoided loading {len(batch_files)} files!)")
    else:
        # Fallback: Load files to count (SLOW - only for old batch directories)
        print(f"   ⚠ Batch sizes metadata not found, loading files to count (slow)...")
        total_samples = 0
        for batch_file in batch_files:
            batch_data = np.load(batch_file)
            total_samples += len(batch_data['features'])
            batch_data.close()

    print(f"   Total samples to merge: {total_samples:,}")

    # Get shapes from first batch
    first_batch = np.load(batch_files[0])
    n_features = first_batch['features'].shape[2]
    first_batch.close()

    # Calculate memory requirements (bool masks = 1 byte, float32 = 4 bytes)
    features_size_gb = total_samples * seq_length * n_features * 4 / 1e9
    masks_size_gb = total_samples * seq_length * n_features * 1 / 1e9  # bool = 1 byte!
    print(f"   Dataset size: {features_size_gb:.1f}GB features + {masks_size_gb:.1f}GB masks = {features_size_gb + masks_size_gb:.1f}GB total")
    print(f"   Masks using bool dtype (75% smaller than float32!)")
    print(f"   Using disk-backed memory-mapped arrays (won't consume RAM)")

    # Create memory-mapped arrays on disk (these don't consume RAM!)
    print(f"   Pass 2/2: Creating memory-mapped arrays and copying batch data...")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if ALL required memmap files exist (dates are not memmap - saved with np.save)
    required_files = ["features.npy", "targets.npy", "masks.npy", "target_stations.npy",
                      "skip_pattern.npy", "permutation.npy"]
    all_files_exist = all((output_path / f).exists() for f in required_files)

    mode = 'r+' if all_files_exist else 'w+'
    print(f"   Mode: {'Updating existing' if mode == 'r+' else 'Creating new'} memmap files")

    # Use np.lib.format.open_memmap() to create proper .npy files with headers
    # (unlike np.memmap which creates raw binary files that can't be loaded with np.load)
    all_features = np.lib.format.open_memmap(
        str(output_path / "features.npy"), dtype=np.float32, mode=mode,
        shape=(total_samples, seq_length, n_features)
    )
    all_targets = np.lib.format.open_memmap(
        str(output_path / "targets.npy"), dtype=np.float32, mode=mode,
        shape=(total_samples, 1)
    )
    all_masks = np.lib.format.open_memmap(
        str(output_path / "masks.npy"), dtype=bool, mode=mode,
        shape=(total_samples, seq_length, n_features)
    )
    all_target_stations = np.lib.format.open_memmap(
        str(output_path / "target_stations.npy"), dtype=np.int32, mode=mode,
        shape=(total_samples,)
    )
    all_skip_pattern = np.lib.format.open_memmap(
        str(output_path / "skip_pattern.npy"), dtype=np.int32, mode=mode,
        shape=(total_samples,)
    )
    all_permutation = np.lib.format.open_memmap(
        str(output_path / "permutation.npy"), dtype=np.int32, mode=mode,
        shape=(total_samples,)
    )

    # Use regular arrays for dates (small enough - ~256MB for 16M samples)
    # Will save with np.save() at the end to match non-augmented dataset format
    all_end_dates = np.zeros(total_samples, dtype=np.float64)
    all_start_dates = np.zeros(total_samples, dtype=np.float64)

    if mode == 'w+':
        # Copy batch data into memory-mapped arrays
        current_idx = 0
        for i, batch_file in enumerate(tqdm(batch_files, desc="      Copying batches", unit="batch")):
            batch_data = np.load(batch_file)

            batch_size = len(batch_data['features'])
            end_idx = current_idx + batch_size

            # Copy into memory-mapped arrays (writes to disk, not RAM)
            all_features[current_idx:end_idx] = batch_data['features']
            all_targets[current_idx:end_idx] = batch_data['targets']
            all_masks[current_idx:end_idx] = batch_data['masks']
            all_target_stations[current_idx:end_idx] = batch_data['target_stations']
            all_end_dates[current_idx:end_idx] = batch_data['end_dates']
            all_start_dates[current_idx:end_idx] = batch_data['start_dates']
            all_skip_pattern[current_idx:end_idx] = batch_data['skip_pattern']
            all_permutation[current_idx:end_idx] = batch_data['permutation']

            # CRITICAL: Close and delete batch immediately to free memory
            batch_data.close()
            del batch_data

            # Flush to disk periodically to avoid buffer buildup
            if (i + 1) % 50 == 0:
                all_features.flush()
                all_targets.flush()
                all_masks.flush()

            current_idx = end_idx

        # Final flush to ensure all data is written to disk
        print(f"   Flushing data to disk...")
        all_features.flush()
        all_targets.flush()
        all_masks.flush()
        all_target_stations.flush()
        all_skip_pattern.flush()
        all_permutation.flush()

    print(f"   Merge complete: {total_samples:,} samples")

    # Normalize
    print(f"\n6. Computing normalization statistics...")

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

            # Mask is now boolean, no need for > 0 comparison
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

    # Normalize in batches (working with memory-mapped arrays)
    # VECTORIZED VERSION - much faster than sample-by-sample loops!
    print(f"\n7. Normalizing augmented samples (vectorized)...")

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
    print(f"\n8. Flushing all changes to disk...")
    all_features.flush()
    all_targets.flush()
    all_masks.flush()
    all_target_stations.flush()
    all_skip_pattern.flush()
    all_permutation.flush()

    # Save dates with np.save() (matches non-augmented dataset format)
    np.save(output_path / 'end_dates.npy', all_end_dates)
    np.save(output_path / 'start_dates.npy', all_start_dates)
    np.save(output_path / "is_normalized.npy", np.array([True], dtype=bool))

    np.savez(
        norm_stats_path,
        feature_mins=feature_mins,
        feature_maxs=feature_maxs,
        target_min=target_min,
        target_max=target_max
    )

    # Cleanup temporary files
    print(f"\n10. Cleaning up temporary files...")
    print(f"   Removing batch files...")
    shutil.rmtree(batch_dir)

    print(f"\n   Saved to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1e9:.2f} GB")

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

    # Choose mode based on memory constraints
    if len(sys.argv) > 1 and sys.argv[1] == "--sequential":
        # Sequential mode: ~5GB RAM (slow but minimal memory)
        print("Using SEQUENTIAL mode (minimal memory)")
        generate_all_augmentations_sequential()
    else:
        # Batched mode: auto-detects CPU cores (faster)
        print("Using BATCHED mode (parallel, auto-detect workers)")
        print("Tip: Use --sequential for systems with <8GB RAM")
        generate_all_augmentations_batched(batch_size=100)
