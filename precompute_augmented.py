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


# Global variable to hold the dataset in each worker process
_worker_dataset = None


def _init_worker(dataset_params, aug_params):
    """
    Initializer function called once when each worker process starts.
    Loads the dataset once per worker (not per batch!).

    Args:
        dataset_params: Tuple of (timeseries_path, stations_path, nearest_path, dense_path)
        aug_params: Dict with augmentation parameters including dimensions and filtered_params
    """
    global _worker_dataset

    timeseries_path, stations_path, nearest_path, dense_path = dataset_params
    seq_length, n_nearby_available, _ = aug_params['dimensions']
    filtered_params = aug_params['filtered_params']

    # Load dataset ONCE for this worker
    _worker_dataset = SoilMoistureSequenceDataset(
        timeseries=str(timeseries_path),
        stations=str(stations_path),
        nearest=str(nearest_path),
        seq_length=seq_length,
        n_nearest=n_nearby_available,
        feature_params=filtered_params,
        dense_array_path=str(dense_path) if Path(dense_path).exists() else None,
        normalize=False
    )


def _process_batch_worker(batch_info):
    """
    Worker function for parallel batch processing.
    Uses the globally loaded dataset (loaded once per worker).

    Args:
        batch_info: Tuple of (batch_num, start_idx, end_idx, aug_params, batch_dir)

    Returns:
        Tuple of (batch_file_path, batch_size) to avoid re-loading later
    """
    global _worker_dataset
    batch_num, start_idx, end_idx, aug_params, batch_dir = batch_info

    # Unpack parameters
    seq_length, n_nearby_available, n_nearby_in_features = aug_params['dimensions']
    skip_patterns = aug_params['skip_patterns']
    all_permutations = aug_params['permutations']
    total_augmentations = aug_params['total_augmentations']
    target_features = aug_params['target_features']
    nearby_features_per_station = aug_params['nearby_features_per_station']
    total_features = aug_params['total_features']

    # Use the pre-loaded dataset
    dataset = _worker_dataset

    batch_actual_size = end_idx - start_idx
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

    for base_idx in range(start_idx, end_idx):
        # Get base sample
        sample = dataset[base_idx]
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

                sample_info = dataset.sample_index[base_idx]
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


def generate_all_augmentations_batched(
    data_dir: str = "./meteogalicia_data",
    n_nearby_available: int = 5,
    n_nearby_in_features: int = 4,
    coverage_threshold: float = 0.25,
    seq_length: int = 64,
    batch_size: int = 1000,  # Process 100 base samples at a time
    num_workers: int = 15  # Number of worker processes (None = min(8, cpu_count))
):
    """
    Pre-compute ALL augmented samples with batched processing (memory efficient!)

    Uses multiprocessing with worker initializers to parallelize augmentation:
    - Each worker loads the dataset ONCE when it starts
    - Workers then process multiple batches without reloading
    - Memory usage: ~2-3GB per worker (instead of per batch!)

    Args:
        data_dir: Directory containing the MeteoGalicia dataset
        n_nearby_available: Number of nearby stations in base dataset (5)
        n_nearby_in_features: Number of nearby stations in augmented samples (4)
        coverage_threshold: Minimum coverage to include a parameter (0.25 = 25%)
        batch_size: Number of base samples to process per batch
        num_workers: Number of parallel workers (default: min(8, cpu_count) for memory efficiency)
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

    # Load base dataset with n_nearest=5
    print(f"\n2. Loading base dataset with {n_nearby_available} nearby stations...")
    if not batch_dir.exists():
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


        # Use conservative default for workers to avoid excessive memory usage
        # Each worker loads ~2-3GB, so 8 workers = ~16-24GB total
        if num_workers is None:
            num_workers = min(8, mp.cpu_count())

        print(f"\n4. Processing in batches of {batch_size} using {num_workers} CPU cores...")
        print(f"   Batch directory: {batch_dir}")
        print(f"   Total batches: {num_batches}")
        print(f"   Estimated memory per worker: ~2-3GB")
        print(f"   Total estimated memory: ~{num_workers * 2.5:.1f}GB")
        print(f"   Estimated speedup: ~{num_workers}x faster than sequential")

        # Prepare parameters for workers
        dataset_params = (collector.timeseries_file, collector.stations_file, collector.nearest_file, dense_path)

        # Process batches in parallel with worker initializer
        print(f"   Starting parallel processing...")
        print(f"   Each worker will load the dataset once, then process multiple batches...")
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(dataset_params, aug_params)) as pool:
            # Use imap with tqdm for progress tracking
            batch_files = []
            batch_sizes = []
            for batch_file, batch_size in tqdm(pool.imap(_process_batch_worker, batch_infos),
                                                total=num_batches,
                                                desc="      Processing batches",
                                                unit="batch"):
                batch_files.append(batch_file)
                batch_sizes.append(batch_size)

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

    # Check if ALL required memmap files exist (not just features.npy!)
    required_files = ["features.npy", "targets.npy", "masks.npy", "target_stations.npy",
                      "skip_pattern.npy", "permutation.npy", "end_dates.npy", "start_dates.npy"]
    all_files_exist = all((output_path / f).exists() for f in required_files)

    mode = 'r+' if all_files_exist else 'w+'
    print(f"   Mode: {'Updating existing' if mode == 'r+' else 'Creating new'} memmap files")

    all_features = np.memmap(
        str(output_path / "features.npy"), dtype=np.float32, mode=mode,
        shape=(total_samples, seq_length, n_features)
    )
    all_targets = np.memmap(
        str(output_path / "targets.npy"), dtype=np.float32, mode=mode,
        shape=(total_samples, 1)
    )
    all_masks = np.memmap(
        str(output_path / "masks.npy"), dtype=bool, mode=mode,
        shape=(total_samples, seq_length, n_features)
    )
    all_target_stations = np.memmap(
        str(output_path / "target_stations.npy"), dtype=np.int32, mode=mode,
        shape=(total_samples,)
    )
    all_skip_pattern = np.memmap(
        str(output_path / "skip_pattern.npy"), dtype=np.int32, mode=mode,
        shape=(total_samples,)
    )
    all_permutation = np.memmap(
        str(output_path / "permutation.npy"), dtype=np.int32, mode=mode,
        shape=(total_samples,)
    )
    all_end_dates = np.memmap(
        str(output_path / "end_dates.npy"), dtype=np.float64, mode=mode,
        shape=(total_samples,)
    )
    all_start_dates = np.memmap(
        str(output_path / "start_dates.npy"), dtype=np.float64, mode=mode,
        shape=(total_samples,)
    )

    if mode == 'w+':
        # Copy batch data into memory-mapped arrays
        current_idx = 0
        for batch_file in tqdm(batch_files, desc="      Copying batches", unit="batch"):
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
    all_end_dates.flush()
    all_start_dates.flush()
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
    print(f"  - Per batch processing: ~{batch_size * total_augmentations * seq_length * total_features * 4 / 1e9:.1f} GB")
    print(f"  - Total worker memory: ~{num_workers * 2.5:.1f} GB")
    print(f"  - Merge/normalize: <5 GB (used memory-mapped arrays)")
    print(f"  - Disk space used: ~{(features_size_gb + masks_size_gb):.1f} GB temporary + {output_path.stat().st_size / 1e9:.1f} GB final")
    print("=" * 70)


if __name__ == "__main__":
    generate_all_augmentations_batched(batch_size=100)
