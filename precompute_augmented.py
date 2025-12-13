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
from pathlib import Path
from itertools import permutations
from Moisturizer import expand_canonical_to_augmented_stats, FeatureLayout
from MeteoGaliciaCollector import MeteoGaliciaCollector
from SoilMoistureSequenceDataset import SoilMoistureSequenceDataset
import multiprocessing as mp
from tqdm import tqdm
from typing import List


def build_skip_patterns(n_nearby_available: int, n_nearby_in_features: int) -> List[List[int]]:
    """
    Build skip patterns for augmentation.

    When n_nearby_available > n_nearby_in_features, we can generate multiple
    augmentations by dropping one station at a time. Each skip pattern contains
    the indices of stations to keep.

    Args:
        n_nearby_available: Number of nearby stations in base dataset (e.g., 5)
        n_nearby_in_features: Number of nearby stations in output (e.g., 4)

    Returns:
        List of index lists, where each list contains indices of stations to keep.
        - If n_nearby_available > n_nearby_in_features: returns n_nearby_available patterns
        - If n_nearby_available == n_nearby_in_features: returns single pattern with all indices
    """
    available_indices = list(range(n_nearby_available))
    skip_patterns = []
    if n_nearby_available > n_nearby_in_features:
        # We can skip one station and still have enough
        for skip_idx in range(n_nearby_available):
            keep_indices = [i for i in available_indices if i != skip_idx][:n_nearby_in_features]
            skip_patterns.append(keep_indices)
    else:
        # n_nearby_available == n_nearby_in_features: use all stations (no skipping)
        skip_patterns.append(list(range(n_nearby_in_features)))
    return skip_patterns


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
                    # FULLY VECTORIZED normalization (100x faster than loop!)
                    # Create invalid mask for entire array at once
                    invalid_mask = np.isin(aug_features, invalid_markers)

                    # Broadcast normalize all features at once
                    # aug_features: (seq_length, total_features)
                    # feature_mins/maxs: (total_features,)
                    feat_ranges = feature_maxs - feature_mins
                    valid_ranges = feat_ranges > 0

                    # Normalize all features in one operation
                    aug_features = 2.0 * (aug_features - feature_mins[None, :]) / np.where(valid_ranges[None, :], feat_ranges[None, :], 1.0) - 1.0

                    # Set invalid values
                    aug_features[invalid_mask] = normalized_invalid_marker

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

    # Return batch info and dates (dates are small, safe to pickle)
    return (batch_num, start_idx, np.array(batch_end_dates, dtype=np.float64),
            np.array(batch_start_dates, dtype=np.float64))


def _setup_augmentation(data_dir: str, n_nearby_available: int, n_nearby_in_features: int,
                        coverage_threshold: float, seq_length: int):
    """
    Common setup for augmentation generation.

    Returns:
        dict with keys: collector, output_path, base_dataset,
                        filtered_params, skip_patterns, all_permutations,
                        total_augmentations, layout, total_samples, base_stats
    """
    collector = MeteoGaliciaCollector(data_dir=data_dir)
    output_path = Path(data_dir) / "precomputed_sequences_augmented"

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
    skip_patterns = build_skip_patterns(n_nearby_available, n_nearby_in_features)
    all_permutations = list(permutations(range(n_nearby_in_features)))
    total_augmentations = len(skip_patterns) * len(all_permutations)

    print(f"   Skip patterns: {len(skip_patterns)}")
    print(f"   Permutations per skip: {len(all_permutations)}")
    print(f"   Total augmentations per base: {total_augmentations}")

    # Calculate dimensions using FeatureLayout
    layout = FeatureLayout(n_params=len(filtered_params), n_nearby=n_nearby_in_features)
    total_samples = len(base_dataset.sample_index) * total_augmentations

    print(f"   Total augmented samples: {total_samples:,}")
    print(f"   Sample shape: [{seq_length}, {layout.n_total_features}]")

    # Calculate memory requirements
    features_size_gb = total_samples * seq_length * layout.n_total_features * 4 / 1e9
    masks_size_gb = total_samples * seq_length * layout.n_total_features * 1 / 1e9
    print(f"   Dataset size: {features_size_gb:.1f}GB features + {masks_size_gb:.1f}GB masks = {features_size_gb + masks_size_gb:.1f}GB total")

    # Load normalization stats from base dataset
    print(f"\n4. Loading normalization stats...")
    canonical_stats_path = Path(data_dir) / "normalization_stats.npz"

    if not canonical_stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found: {canonical_stats_path}\n"
            f"Run dataset build first with buildDataset() to create normalization stats."
        )

    print(f"   Loading from {canonical_stats_path}...")
    stats = np.load(canonical_stats_path, allow_pickle=True)

    if 'target_feature_mins' not in stats:
        raise ValueError(
            "Stats file missing 'target_feature_mins' (old format not supported). "
            "Regenerate the base dataset with buildDataset() to create new format stats."
        )

    base_stats = expand_canonical_to_augmented_stats(
        canonical_stats=stats,
        n_params=len(filtered_params),
        n_nearby_in_features=n_nearby_in_features,
        n_nearby_available=n_nearby_available,
        augmented=True
    )

    n_samples = int(stats['n_base_samples'][0]) if 'n_base_samples' in stats else 0
    print(f"   ✓ Loaded canonical stats ({n_samples:,} samples)")
    print(f"   Feature range: [{base_stats['feature_mins'].min():.2f}, {base_stats['feature_maxs'].max():.2f}]")
    print(f"   Target range: [{base_stats['target_min']:.2f}, {base_stats['target_max']:.2f}]")

    return {
        'collector': collector,
        'output_path': output_path,
        'base_dataset': base_dataset,
        'filtered_params': filtered_params,
        'skip_patterns': skip_patterns,
        'all_permutations': all_permutations,
        'total_augmentations': total_augmentations,
        'layout': layout,
        'total_samples': total_samples,
        'base_stats': base_stats,
        'seq_length': seq_length,
    }


def _create_memmap_arrays(output_path, total_samples, seq_length, total_features):
    """
    Create memory-mapped arrays for augmented dataset.

    Returns:
        dict with keys: features, targets, masks, target_stations, skip_pattern,
                        permutation, end_dates, start_dates
    """
    output_path.mkdir(parents=True, exist_ok=True)

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

    # Regular arrays for dates (will be saved with np.save)
    all_end_dates = np.empty(total_samples, dtype=np.float64)
    all_start_dates = np.empty(total_samples, dtype=np.float64)

    return {
        'features': all_features,
        'targets': all_targets,
        'masks': all_masks,
        'target_stations': all_target_stations,
        'skip_pattern': all_skip_pattern,
        'permutation': all_permutation,
        'end_dates': all_end_dates,
        'start_dates': all_start_dates,
    }


def _save_augmented_dataset(arrays, output_path):
    """
    Flush memmap arrays and save dataset metadata.

    Note: Normalization stats are NOT saved here - the augmented dataset uses
    the canonical stats from normalization_stats.npz which are expanded at load time.
    """
    arrays['features'].flush()
    arrays['targets'].flush()
    arrays['masks'].flush()
    arrays['target_stations'].flush()
    arrays['skip_pattern'].flush()
    arrays['permutation'].flush()

    np.save(output_path / 'end_dates.npy', arrays['end_dates'])
    np.save(output_path / 'start_dates.npy', arrays['start_dates'])
    np.save(output_path / 'is_normalized.npy', np.array([True], dtype=bool))


def _print_completion_stats(base_dataset, total_samples, num_workers=None, disk_size_gb=0):
    """
    Print completion statistics.
    """
    print("\n" + "=" * 70)
    print("✓ AUGMENTED DATASET COMPLETE!")
    print("=" * 70)
    print(f"Base samples: {len(base_dataset.sample_index):,}")
    print(f"Augmented samples: {total_samples:,}")
    print(f"Augmentation factor: {total_samples / len(base_dataset.sample_index):.0f}x")
    print(f"\nMemory usage:")
    if num_workers:
        print(f"  - Dataset (main process): ~4 GB")
        print(f"  - Workers: ~{num_workers * 0.5:.1f} GB")
        print(f"  - Total peak RAM: ~{4 + num_workers * 0.5:.1f} GB")
    else:
        print(f"  - Peak RAM: ~5 GB (sequential processing)")
    print(f"  - Disk space: {disk_size_gb:.1f} GB")
    print(f"\nPerformance:")
    print(f"  - Normalization: Done during generation using base stats")
    print("=" * 70)


def generate_all_augmentations_sequential(
    data_dir: str = "./meteogalicia_data",
    n_nearby_available: int = 5,
    n_nearby_in_features: int = 4,
    coverage_threshold: float = 0.25,
    seq_length: int = 64,
):
    """
    Pre-compute ALL augmented samples SEQUENTIALLY (minimal memory!)

    No multiprocessing, no temp batches - writes directly to memory-mapped arrays.
    Uses base dataset normalization stats for accurate and fast normalization.

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

    # Use shared setup function
    setup = _setup_augmentation(
        data_dir, n_nearby_available, n_nearby_in_features,
        coverage_threshold, seq_length
    )

    base_dataset = setup['base_dataset']
    output_path = setup['output_path']
    skip_patterns = setup['skip_patterns']
    all_permutations = setup['all_permutations']
    total_augmentations = setup['total_augmentations']
    layout = setup['layout']
    total_samples = setup['total_samples']
    base_stats = setup['base_stats']
    filtered_params = setup['filtered_params']

    target_features = layout.n_target_features
    nearby_features_per_station = layout.nearby_features_per_station
    total_features = layout.n_total_features

    # Create memory-mapped arrays
    print(f"\n5. Creating memory-mapped arrays...")
    arrays = _create_memmap_arrays(output_path, total_samples, seq_length, total_features)

    # Process samples sequentially
    print(f"\n6. Generating augmentations (sequential)...")
    current_idx = 0

    # Normalization params if using base stats
    invalid_markers = [-9999.0, -1000.0]
    normalized_invalid_marker = -2.0

    for base_idx in tqdm(range(len(base_dataset.sample_index)), desc="   Processing", unit="sample"):
        # Get base sample
        sample = base_dataset[base_idx]
        base_features = sample['features'].numpy()
        base_mask = sample['mask'].numpy()
        base_target = sample['target'].numpy()[0]
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

                # Vectorized normalization using base stats
                invalid_mask = np.isin(aug_features, invalid_markers)

                feature_mins_arr = base_stats['feature_mins']
                feature_maxs_arr = base_stats['feature_maxs']
                feat_ranges = feature_maxs_arr - feature_mins_arr
                valid_ranges = feat_ranges > 0

                aug_features = 2.0 * (aug_features - feature_mins_arr[None, :]) / np.where(valid_ranges[None, :], feat_ranges[None, :], 1.0) - 1.0
                aug_features[invalid_mask] = normalized_invalid_marker

                # Normalize target
                if base_target not in invalid_markers:
                    target_min = base_stats['target_min']
                    target_max = base_stats['target_max']
                    if target_max > target_min:
                        normalized_target = 2.0 * (base_target - target_min) / (target_max - target_min) - 1.0
                    else:
                        normalized_target = base_target
                else:
                    normalized_target = normalized_invalid_marker

                # Write directly to memmap (already normalized)
                arrays['features'][current_idx] = aug_features
                arrays['targets'][current_idx] = normalized_target
                arrays['masks'][current_idx] = aug_mask
                arrays['target_stations'][current_idx] = sample_info['target_station']
                arrays['end_dates'][current_idx] = sample_info['end_date'].timestamp()
                arrays['start_dates'][current_idx] = sample_info['start_date'].timestamp()
                arrays['skip_pattern'][current_idx] = skip_idx
                arrays['permutation'][current_idx] = perm_idx

                current_idx += 1

    print(f"   ✓ Generated {current_idx:,} augmented samples")

    # Save dataset using helper function
    print(f"\n7. Saving dataset...")
    _save_augmented_dataset(arrays, output_path)
    print(f"   ✓ Saved to: {output_path}")

    # Print completion stats
    features_size_gb = total_samples * seq_length * total_features * 4 / 1e9
    masks_size_gb = total_samples * seq_length * total_features * 1 / 1e9
    _print_completion_stats(base_dataset, total_samples, num_workers=None,
                           disk_size_gb=features_size_gb + masks_size_gb)


def generate_all_augmentations_batched(
    data_dir: str = "./meteogalicia_data",
    n_nearby_available: int = 5,
    n_nearby_in_features: int = 4,
    coverage_threshold: float = 0.25,
    seq_length: int = 64,
    batch_size: int = 1000,
    num_workers: int = None,  # Auto-detect: physical cores (avoids hyperthreading)
):
    """
    Pre-compute ALL augmented samples with batched processing (memory efficient!)

    Uses multiprocessing with PARALLEL FETCHING:
    - Each worker loads its own dataset instance (memory-mapped arrays are shared!)
    - Workers fetch samples in parallel (no blocking on main process)
    - Memory usage: ~4GB (shared memmaps) + ~100MB per worker (metadata only)
    - Uses base dataset normalization stats for accurate and fast normalization

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
    skip_patterns = build_skip_patterns(n_nearby_available, n_nearby_in_features)
    all_permutations = list(permutations(range(n_nearby_in_features)))
    total_augmentations = len(skip_patterns) * len(all_permutations)

    print(f"   Skip patterns: {len(skip_patterns)}")
    print(f"   Permutations per skip: {len(all_permutations)}")
    print(f"   Total augmentations per base: {total_augmentations}")

    # Create batch info tuples for all batches
    # Note: No longer passing dataset_params in batch_info since dataset is loaded in initializer
    layout = FeatureLayout(n_params=len(filtered_params), n_nearby=n_nearby_in_features)
    target_features = layout.n_target_features
    nearby_features_per_station = layout.nearby_features_per_station
    total_features = layout.n_total_features
    aug_params = {
        'dimensions': (seq_length, n_nearby_available, n_nearby_in_features),
        'skip_patterns': skip_patterns,
        'permutations': all_permutations,
        'filtered_params': filtered_params,
        'total_augmentations': total_augmentations,
        'target_features': target_features,
        'nearby_features_per_station': nearby_features_per_station,
        'total_features': total_features,
        'debug': False  # Disabled - debugging complete
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

    # Load normalization stats from base dataset
    print(f"\n4. Loading normalization stats...")
    canonical_stats_path = Path(data_dir) / "normalization_stats.npz"

    if not canonical_stats_path.exists():
        raise FileNotFoundError(
            f"Stats file not found: {canonical_stats_path}\n"
            f"Run dataset build first with buildDataset() to create normalization stats."
        )

    print(f"   Loading from {canonical_stats_path}...")
    stats = np.load(canonical_stats_path, allow_pickle=True)

    # Check if canonical format is available
    if 'target_feature_mins' not in stats:
        raise ValueError(
            "Stats file missing 'target_feature_mins' (old format not supported). "
            "Regenerate the base dataset with buildDataset() to create new format stats."
        )

    # Use the shared function to expand canonical stats
    base_stats = expand_canonical_to_augmented_stats(
        canonical_stats=stats,
        n_params=len(filtered_params),
        n_nearby_in_features=n_nearby_in_features,
        n_nearby_available=n_nearby_available,
        augmented=True
    )

    n_samples = int(stats['n_base_samples'][0]) if 'n_base_samples' in stats else 0
    print(f"   ✓ Loaded canonical stats ({n_samples:,} samples)")
    print(f"   Feature range: [{base_stats['feature_mins'].min():.2f}, {base_stats['feature_maxs'].max():.2f}]")
    print(f"   Target range: [{base_stats['target_min']:.2f}, {base_stats['target_max']:.2f}]")

    # Add stats to aug_params for workers
    aug_params['normalize'] = True
    aug_params['feature_mins'] = base_stats['feature_mins']
    aug_params['feature_maxs'] = base_stats['feature_maxs']
    aug_params['target_min'] = float(base_stats['target_min'])
    aug_params['target_max'] = float(base_stats['target_max'])
    aug_params['invalid_markers'] = [-9999.0, -1000.0]

    print(f"\n5. Creating memmap files and processing in parallel (direct write, no serialization!)...")
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

            # Fetch samples for this batch (main process)
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
    print(f"\n6. Data written to memmap files:")
    print(f"   Dataset size: {features_size_gb:.1f}GB features + {masks_size_gb:.1f}GB masks = {features_size_gb + masks_size_gb:.1f}GB total")
    print(f"   Masks using bool dtype (75% smaller than float32!)")

    # Save normalization flag (data already normalized by workers)
    # Note: We don't save augmented stats - the canonical stats from normalization_stats.npz
    # are expanded at load time by expand_canonical_to_augmented_stats()
    np.save(output_path / "is_normalized.npy", np.array([True], dtype=bool))
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
    print(f"  - Normalization: Done in workers using base stats")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Pre-compute augmented dataset for soil moisture prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python precompute_augmented.py                          # Batched mode (default, fast)
  python precompute_augmented.py --sequential             # Sequential mode (low memory)
  python precompute_augmented.py --seq-length 32          # Use sequence length of 32
        """
    )
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential mode (~5GB RAM, slower but minimal memory)')
    parser.add_argument('--seq-length', type=int, default=64,
                       help='Sequence length (default: 64)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for batched mode (default: 100)')

    args = parser.parse_args()

    # Choose mode based on arguments
    if args.sequential:
        print("Using SEQUENTIAL mode (minimal memory)")
        print(f"Sequence length: {args.seq_length}")
        generate_all_augmentations_sequential(seq_length=args.seq_length)
    else:
        print("Using BATCHED mode (parallel, auto-detect workers)")
        print(f"Sequence length: {args.seq_length}")
        print(f"Batch size: {args.batch_size}")
        generate_all_augmentations_batched(
            batch_size=args.batch_size,
            seq_length=args.seq_length
        )
