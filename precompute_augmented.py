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
        Path to saved batch file
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
    batch_masks = np.zeros((batch_aug_size, seq_length, total_features), dtype=np.float32)
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

    return batch_file


def generate_all_augmentations_batched(
    data_dir: str = "./meteogalicia_data",
    n_nearby_available: int = 5,
    n_nearby_in_features: int = 4,
    coverage_threshold: float = 0.25,
    batch_size: int = 100,  # Process 100 base samples at a time
    num_workers: int = None  # Number of worker processes (None = min(8, cpu_count))
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

    # Create temporary directory for batch files under data_dir (not /tmp)
    batch_dir = Path(data_dir) / "augmented_batches"
    batch_dir.mkdir(exist_ok=True)
    num_batches = (len(base_dataset.sample_index) + batch_size - 1) // batch_size

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

    # Create batch info tuples for all batches
    # Note: No longer passing dataset_params in batch_info since dataset is loaded in initializer
    batch_infos = []
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(base_dataset.sample_index))
        batch_infos.append((batch_num, start_idx, end_idx, aug_params, str(batch_dir)))

    # Process batches in parallel with worker initializer
    print(f"   Starting parallel processing...")
    print(f"   Each worker will load the dataset once, then process multiple batches...")
    with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(dataset_params, aug_params)) as pool:
        # Use imap to get progress updates as batches complete
        batch_files = []
        for i, batch_file in enumerate(pool.imap(_process_batch_worker, batch_infos)):
            batch_files.append(batch_file)
            if (i + 1) % max(1, num_batches // 20) == 0 or (i + 1) == num_batches:
                print(f"      Progress: {i+1}/{num_batches} batches complete ({100*(i+1)/num_batches:.1f}%)")

    print(f"   All {num_batches} batches processed!")

    # Merge all batches - MEMORY EFFICIENT VERSION
    print(f"\n5. Merging {len(batch_files)} batches...")
    print(f"   This will take a few minutes...")

    # First pass: Calculate total size
    print(f"   Pass 1/2: Calculating total size...")
    total_samples = 0
    for batch_file in batch_files:
        batch_data = np.load(batch_file)
        total_samples += len(batch_data['features'])
        batch_data.close()

    print(f"   Total samples to merge: {total_samples:,}")

    # Get shapes from first batch
    first_batch = np.load(batch_files[0])
    seq_length = first_batch['features'].shape[1]
    n_features = first_batch['features'].shape[2]
    first_batch.close()

    # Pre-allocate final arrays
    print(f"   Pre-allocating final arrays...")
    all_features = np.zeros((total_samples, seq_length, n_features), dtype=np.float32)
    all_targets = np.zeros((total_samples, 1), dtype=np.float32)
    all_masks = np.zeros((total_samples, seq_length, n_features), dtype=np.float32)
    all_target_stations = np.zeros(total_samples, dtype=np.int32)
    all_end_dates = []
    all_start_dates = []
    all_skip_pattern = np.zeros(total_samples, dtype=np.int32)
    all_permutation = np.zeros(total_samples, dtype=np.int32)

    # Second pass: Copy data incrementally
    print(f"   Pass 2/2: Copying batch data...")
    current_idx = 0
    for i, batch_file in enumerate(batch_files):
        if i % 10 == 0:
            print(f"      Batch {i+1}/{len(batch_files)} (progress: {current_idx:,}/{total_samples:,})...")
        batch_data = np.load(batch_file)

        batch_size = len(batch_data['features'])
        end_idx = current_idx + batch_size

        # Copy into pre-allocated arrays
        all_features[current_idx:end_idx] = batch_data['features']
        all_targets[current_idx:end_idx] = batch_data['targets']
        all_masks[current_idx:end_idx] = batch_data['masks']
        all_target_stations[current_idx:end_idx] = batch_data['target_stations']
        all_end_dates.extend(batch_data['end_dates'])
        all_start_dates.extend(batch_data['start_dates'])
        all_skip_pattern[current_idx:end_idx] = batch_data['skip_pattern']
        all_permutation[current_idx:end_idx] = batch_data['permutation']

        # CRITICAL: Close and delete batch immediately to free memory
        batch_data.close()
        del batch_data

        current_idx = end_idx

    print(f"   Merge complete: {total_samples:,} samples")

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
    print(f"\n9. Cleaning up batch files...")
    shutil.rmtree(batch_dir)

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
