#!/usr/bin/env python3

import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import permutations
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import os

from Moisturizer import (
    expand_canonical_to_augmented_stats,
    normalize_features,
    normalize_target,
    FeatureLayout,
    NORMALIZED_INVALID_MARKER,
)


class AugmentedLiveDataset(Dataset):
    """
    Efficient live augmentation dataset that wraps a base dataset with 1 more station.
    The augmentation factor is the factorial of the number of n_nearby_in_features + 1.
    Ie if using 4 nearby stations the augmentation is 5!=120x. With 5 = 720x
    Compatible with train_val_test_split through index mapping.

    There are two ways to create this dataset:

    Option 1: From SoilMoistureSequenceDataset (uses dense arrays, no precompute needed)
        dataset = AugmentedLiveDataset.from_base_dataset(
            timeseries="meteogalicia_data/raw_timeseries.csv",
            stations="meteogalicia_data/stations.csv",
            nearest="meteogalicia_data/nearest.csv",
            dense_array_path="meteogalicia_data/dense_features.npz",
            feature_params=filtered_params
        )

    Option 2: From precomputed base dataset (fastest)
        dataset = AugmentedLiveDataset(
            base_precomputed_path="meteogalicia_data/precomputed_sequences_5nearby"
        )
    """

    def __init__(
        self,
        base_precomputed_path: str,
        norm_stats_path: Optional[str] = None,
        n_nearby_available: int = 5,
        n_nearby_in_features: int = 4,
        seq_length: int = 64,
        feature_params: Optional[List[str]] = None,
    ):
        super().__init__()

        self.seq_length = seq_length
        self.n_nearby_available = n_nearby_available
        self.n_nearby_in_features = n_nearby_in_features

        base_path = Path(base_precomputed_path)
        if not base_path.exists():
            raise ValueError(f"Base precomputed path does not exist: {base_path}")

        print("=" * 70)
        print("AUGMENTED LIVE DATASET V2 - Efficient on-the-fly augmentation")
        print("=" * 70)

        # 1. Load precomputed base dataset (memory-mapped for efficiency)
        print(f"\n1. Loading precomputed base dataset from {base_path}...")
        self._load_base_dataset(base_path)

        # 2. Load or infer feature params
        if feature_params is not None:
            self.feature_params = feature_params
        elif (base_path / "feature_params.npy").exists():
            self.feature_params = list(np.load(base_path / "feature_params.npy"))
        else:
            # Infer from feature dimensions
            self.feature_params = [f"param_{i}" for i in range(self._infer_n_params())]

        # 3. Check if data is pre-normalized
        if (base_path / "is_normalized.npy").exists():
            self.is_prenormalized = bool(np.load(base_path / "is_normalized.npy")[0])
        else:
            self.is_prenormalized = False
            print("   WARNING: Base dataset may not be normalized")

        print(f"   Base samples: {self.n_base_samples:,}")
        print(f"   Pre-normalized: {self.is_prenormalized}")

        # 4. Generate augmentation patterns
        print(f"\n2. Generating augmentation patterns...")
        self._build_augmentation_patterns()
        print(f"   Skip patterns: {len(self.skip_patterns)}")
        print(f"   Permutations: {len(self.all_permutations)}")
        print(f"   Total augmentations per sample: {self.total_augmentations}")
        print(f"   Total dataset size: {len(self):,} samples")

        # 5. Build column index mappings for fast slicing
        print(f"\n3. Building column index mappings...")
        self._build_column_indices()
        print(f"   Target features: {self.n_target_features}")
        print(f"   Nearby features per station: {self.nearby_features_per_station}")
        print(f"   Output features: {self.n_output_features}")

        # 6. Index mapping for splits (None = use all)
        self._indices = None
        self._base_indices = None  # Maps to base sample indices (for splits)

        print("\n✓ Augmented live dataset ready!")
        print("=" * 70)

    @classmethod
    def from_base_dataset(
        cls,
        timeseries: str,
        stations: str,
        nearest: str,
        dense_array_path: str,
        feature_params: List[str],
        seq_length: int = 64,
        n_nearby_available: int = 5,
        n_nearby_in_features: int = 4,
        normalize: bool = True,
        norm_stats_path: Optional[str] = None,
        precomputed_path: Optional[str] = None,
    ) -> 'AugmentedLiveDataset':
        """
        Create dataset from raw data files using SoilMoistureSequenceDataset.

        This approach doesn't require a precomputed 5-nearby dataset, but it will
        build the base dataset on initialization (which can take a few minutes).

        The base dataset is built with n_nearby_available nearby stations, then
        augmentations are applied on-the-fly to produce n_nearby_in_features.

        Args:
            timeseries: Path to raw_timeseries.csv
            stations: Path to stations.csv
            nearest: Path to nearest.csv
            dense_array_path: Path to dense_features.npz (required if no precomputed!)
            feature_params: List of feature parameters to include
            seq_length: Sequence length (default 64)
            n_nearby_available: Number of nearby stations to load (default 5)
            n_nearby_in_features: Number of nearby stations in output (default 4)
            normalize: Whether to normalize (default True)
            norm_stats_path: Path to normalization stats file (optional - will compute if not provided)
            precomputed_path: Path to precomputed base dataset (n+1 nearby) - Faster
        """
        # Import here to avoid circular dependency
        from Moisturizer import SoilMoistureSequenceDataset

        print("=" * 70)
        print("AUGMENTED LIVE DATASET V2 - Building from base dataset")
        print("=" * 70)

        # Check if precomputed base dataset exists
        if precomputed_path and Path(precomputed_path).exists():
            print(f"\n1. Loading PRECOMPUTED base dataset from {precomputed_path}...")
            print("   (This is the fast path!)")
            base_dataset = SoilMoistureSequenceDataset(
                timeseries=timeseries,
                stations=stations,
                nearest=nearest,
                seq_length=seq_length,
                n_nearest=n_nearby_available,
                feature_params=feature_params,
                precomputed_path=precomputed_path,
                dense_array_path=dense_array_path,
                normalize=False,  # We'll handle normalization ourselves
                norm_stats_path=None
            )
        else:
            print(f"\n1. Building base dataset with {n_nearby_available} nearby stations...")
            print("   WARNING: Using dense arrays (slower than precomputed)")
            print(f"   Consider precomputing with n_nearest={n_nearby_available} for 10x speedup")
            
            base_dataset = SoilMoistureSequenceDataset(
                timeseries=timeseries,
                stations=stations,
                nearest=nearest,
                seq_length=seq_length,
                n_nearest=n_nearby_available,
                feature_params=feature_params,
                precomputed_path=None,
                dense_array_path=dense_array_path,
                normalize=False,  # We'll handle normalization ourselves
                norm_stats_path=None
            )

        # Create the augmented dataset from the base dataset
        instance = cls._from_soil_moisture_dataset(
            base_dataset,
            n_nearby_available=n_nearby_available,
            n_nearby_in_features=n_nearby_in_features,
        )
        
        # Compute or load normalization stats
        if normalize:
            # Default path for canonical stats
            if norm_stats_path is None:
                base_path = None
                if precomputed_path is not None:
                    base_path = Path(precomputed_path).parent
                if timeseries is not None:
                    base_path = Path(timeseries).parent
                norm_stats_path = base_path / "normalization_stats.npz"

            print(f"\n4. Loading/computing normalization stats...")

            # Try to load canonical stats first
            loaded = False
            if Path(norm_stats_path).exists():
                loaded = instance.load_normalization_stats(norm_stats_path)

            if not loaded:
                raise FileNotFoundError(f"The normalization stats were not found in:{norm_stats_path}")

            instance.normalize = True
            instance.normalized_invalid_marker = -2.0
            print(f"   Feature range: [{instance.feature_mins.min():.2f}, {instance.feature_maxs.max():.2f}]")
            print(f"   Target range: [{instance.target_min:.2f}, {instance.target_max:.2f}]")
        else:
            instance.normalize = False
            instance.feature_mins = None
            instance.feature_maxs = None
            instance.target_min = None
            instance.target_max = None
        
        print("\n✓ Augmented live dataset ready!")
        print("=" * 70)
        
        return instance

    @classmethod
    def _from_soil_moisture_dataset(
        cls,
        base_dataset: 'SoilMoistureSequenceDataset',
        n_nearby_available: int = 5,
        n_nearby_in_features: int = 4,
    ) -> 'AugmentedLiveDataset':
        """Create from an existing SoilMoistureSequenceDataset"""
        instance = cls.__new__(cls)

        instance.seq_length = base_dataset.seq_length
        instance.n_nearby_available = n_nearby_available
        instance.n_nearby_in_features = n_nearby_in_features
        instance.feature_params = base_dataset.feature_params
        instance.is_prenormalized = base_dataset.is_prenormalized

        # Store reference to base dataset (for __getitem__)
        instance._base_dataset = base_dataset
        instance.n_base_samples = len(base_dataset)

        # Copy metadata
        instance.sample_index = base_dataset.sample_index
        instance.target_stations = base_dataset.target_stations

        # We don't have memory-mapped arrays, so set these to None
        instance.base_features = None
        instance.base_targets = None
        instance.base_masks = None
        instance.base_target_stations = None
        instance.base_end_dates = None
        instance.base_start_dates = None

        # Initialize index mapping (None = use all, must be set before __len__ is called)
        instance._indices = None
        instance._base_indices = None

        # For compatibility with code that checks for precomputed_data
        # Point to the base dataset's precomputed data if it exists
        instance.precomputed_data = getattr(base_dataset, 'precomputed_data', None)

        print(f"   Base samples: {instance.n_base_samples:,}")
        print(f"   Pre-normalized: {instance.is_prenormalized}")

        # Generate augmentation patterns
        print(f"\n2. Generating augmentation patterns...")
        instance._build_augmentation_patterns()
        print(f"   Skip patterns: {len(instance.skip_patterns)}")
        print(f"   Permutations: {len(instance.all_permutations)}")
        print(f"   Total augmentations per sample: {instance.total_augmentations}")
        print(f"   Total dataset size: {len(instance):,} samples")

        # Build column indices
        print(f"\n3. Building column index mappings...")
        instance._build_column_indices()
        print(f"   Target features: {instance.n_target_features}")
        print(f"   Nearby features per station: {instance.nearby_features_per_station}")
        print(f"   Output features: {instance.n_output_features}")

        return instance

    def _load_base_dataset(self, base_path: Path):
        """Load base dataset arrays with memory mapping"""
        # Memory-map the large arrays
        self.base_features = np.load(base_path / "features.npy", mmap_mode='r')
        self.base_targets = np.load(base_path / "targets.npy", mmap_mode='r')
        self.base_masks = np.load(base_path / "masks.npy", mmap_mode='r')

        # Load metadata arrays fully (small)
        self.base_target_stations = np.load(base_path / "target_stations.npy")
        self.base_end_dates = np.load(base_path / "end_dates.npy")
        self.base_start_dates = np.load(base_path / "start_dates.npy")

        self.n_base_samples = len(self.base_features)

        # Build sample_index for compatibility with split code
        self.sample_index = []
        for i in range(self.n_base_samples):
            self.sample_index.append({
                'target_station': int(self.base_target_stations[i]),
                'end_date': self.base_end_dates[i],
                'start_date': self.base_start_dates[i]
            })

        # Get unique target stations for split compatibility
        self.target_stations = list(np.unique(self.base_target_stations))

    def _infer_n_params(self) -> int:
        """Infer number of feature parameters from base dataset shape"""
        total_features = self.base_features.shape[2]
        # total = n_params + n_nearby * (1 + n_params + 1)
        # total = n_params + n_nearby * (n_params + 2)
        # total = n_params * (1 + n_nearby) + 2 * n_nearby
        # n_params = (total - 2 * n_nearby) / (1 + n_nearby)
        n_params = (total_features - 2 * self.n_nearby_available) // (1 + self.n_nearby_available)
        return n_params

    def _build_augmentation_patterns(self):
        """Build skip patterns and permutations"""
        available_indices = list(range(self.n_nearby_available))

        # Skip patterns: drop 1 of n_nearby_available stations
        # Only create skip patterns if we have more stations available than needed
        self.skip_patterns = []
        if self.n_nearby_available > self.n_nearby_in_features:
            # We can skip one station and still have enough
            for skip_idx in range(self.n_nearby_available):
                keep_indices = [i for i in available_indices if i != skip_idx][:self.n_nearby_in_features]
                self.skip_patterns.append(keep_indices)
        else:
            # n_nearby_available == n_nearby_in_features: use all stations (no skipping)
            self.skip_patterns.append(list(range(self.n_nearby_in_features)))

        # All permutations of kept stations
        self.all_permutations = list(permutations(range(self.n_nearby_in_features)))

        self.total_augmentations = len(self.skip_patterns) * len(self.all_permutations)

    def _build_column_indices(self):
        """Precompute column indices for efficient slicing"""
        # Use FeatureLayout for consistent dimension calculations
        layout = FeatureLayout(n_params=len(self.feature_params), n_nearby=self.n_nearby_in_features)

        self.n_target_features = layout.n_target_features
        self.nearby_features_per_station = layout.nearby_features_per_station

        # Output dimensions
        self.n_output_features = layout.n_total_features

        # Target columns (unchanged across augmentations)
        self.target_cols = np.arange(self.n_target_features)

        # Precompute column indices for each skip pattern + permutation
        # This is the key optimization: instead of reshaping at runtime,
        # we precompute exactly which columns to select
        self._aug_column_indices = []

        for skip_pattern in self.skip_patterns:
            for perm in self.all_permutations:
                # Build column indices for this augmentation
                cols = list(self.target_cols)

                # Apply skip pattern then permutation
                permuted_stations = [skip_pattern[p] for p in perm]

                for output_slot, source_station in enumerate(permuted_stations):
                    # Source column range for this station in base data
                    source_start = self.n_target_features + (source_station * self.nearby_features_per_station)
                    source_end = source_start + self.nearby_features_per_station

                    cols.extend(range(source_start, source_end))

                self._aug_column_indices.append(np.array(cols, dtype=np.int64))


    def load_normalization_stats(self, path: str) -> bool:
        """
        Load canonical normalization stats and expand to current augmented layout.

        For augmented datasets, we need to aggregate stats across all available slots
        because any slot can receive data from any of the available stations.

        Returns True if stats were loaded successfully, False if incompatible.
        """
        try:
            stats = np.load(path, allow_pickle=True)

            # Check compatibility
            saved_n_params = int(stats['n_params'][0])
            saved_feature_params = list(stats['feature_params'])

            if saved_n_params != len(self.feature_params):
                print(f"   Warning: Stats have {saved_n_params} params, need {len(self.feature_params)}")
                return False

            if saved_feature_params != self.feature_params:
                print(f"   Warning: Feature params don't match")
                return False

            # Check if canonical format is available
            if 'target_feature_mins' not in stats:
                raise ValueError(
                    "Stats file missing 'target_feature_mins' (old format not supported). "
                    "Regenerate the base dataset with buildDataset() to create new format stats."
                )

            # Use the shared function to expand canonical stats
            expanded_stats = expand_canonical_to_augmented_stats(
                canonical_stats=stats,
                n_params=len(self.feature_params),
                n_nearby_in_features=self.n_nearby_in_features,
                n_nearby_available=self.n_nearby_available,
                augmented=True
            )

            self.feature_mins = expanded_stats['feature_mins']
            self.feature_maxs = expanded_stats['feature_maxs']
            self.target_min = expanded_stats['target_min']
            self.target_max = expanded_stats['target_max']

            print(f"   ✓ Loaded stats from {path} ({int(stats['n_base_samples'][0]):,} samples)")
            return True

        except KeyError as e:
            raise ValueError(f"Stats file missing required key {e}. Regenerate with buildDataset().")
        except Exception as e:
            raise ValueError(f"Could not load stats from {path}: {e}")

    def __len__(self) -> int:
        """Total number of augmented samples"""
        if self._base_indices is not None:
            return len(self._base_indices) * self.total_augmentations
        return self.n_base_samples * self.total_augmentations

    def _get_base_and_aug_idx(self, idx: int) -> Tuple[int, int]:
        """Map augmented index to (base_idx, augmentation_idx)"""
        if self._base_indices is not None:
            # This is a split dataset
            # idx goes from 0 to len(split) - 1
            # We need to map to (base sample in original data, augmentation index)
            n_base_in_split = len(self._base_indices)
            base_in_split = idx // self.total_augmentations
            aug_idx = idx % self.total_augmentations
            
            # Map base_in_split to actual base index in original data
            base_idx = self._base_indices[base_in_split]
        else:
            # Full dataset
            base_idx = idx // self.total_augmentations
            aug_idx = idx % self.total_augmentations
        return base_idx, aug_idx

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get an augmented sample using precomputed column indices.

        This is fast because:
        1. Column selection is precomputed
        2. Single numpy advanced indexing operation
        3. Vectorized normalization
        """
        base_idx, aug_idx = self._get_base_and_aug_idx(idx)

        # Get column indices for this augmentation
        col_indices = self._aug_column_indices[aug_idx]

        if self.base_features is not None:
            # Fast path: precomputed arrays
            features = self.base_features[base_idx][:, col_indices].copy()
            mask = self.base_masks[base_idx][:, col_indices].copy()
            target = self.base_targets[base_idx].copy()
            station_id = int(self.base_target_stations[base_idx])
            end_date = float(self.base_end_dates[base_idx])
        else:
            # Slower path: fetch from base dataset
            sample = self._base_dataset[base_idx]
            features_full = sample['features'].numpy()
            mask_full = sample['mask'].numpy()

            features = features_full[:, col_indices].copy()
            mask = mask_full[:, col_indices].copy()
            target = sample['target'].numpy().copy()
            station_id = sample['target_station_id']
            end_date = sample['end_date']

        # Apply normalization if enabled
        if getattr(self, 'normalize', False) and self.feature_mins is not None:
            features, target = self._apply_normalization(features, mask, target)

        return {
            'features': torch.from_numpy(features),
            'target': torch.from_numpy(target) if isinstance(target, np.ndarray) else torch.tensor([target], dtype=torch.float32),
            'mask': torch.from_numpy(mask),
            'target_station_id': station_id,
            'end_date': end_date
        }

    def _apply_normalization(self, features: np.ndarray, mask: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply min-max normalization to features and target.

        Args:
            features: [seq_length, n_features] array
            mask: [seq_length, n_features] boolean array
            target: [1] array or scalar

        Returns:
            normalized_features, normalized_target
        """
        # Normalize features using shared function
        features = normalize_features(
            features, self.feature_mins, self.feature_maxs,
            normalized_invalid_marker=self.normalized_invalid_marker
        )

        # Normalize target using shared function
        normalized_val = normalize_target(
            target, self.target_min, self.target_max,
            normalized_invalid_marker=self.normalized_invalid_marker
        )
        normalized_target = np.array([normalized_val], dtype=np.float32)

        return features.astype(np.float32), normalized_target

    @staticmethod
    def train_val_test_split(
        dataset: 'AugmentedLiveDataset',
        val_stations_ratio: float = 0.15,
        test_stations_ratio: float = 0.0,
        random_seed: int = 42
    ) -> Tuple['AugmentedLiveDataset', Optional['AugmentedLiveDataset'], Optional['AugmentedLiveDataset']]:
        """
        Split dataset by stations (not by time) for better generalization.

        Splits are done at the STATION level, not sample level.
        All samples from a station go to the same split.
        """
        np.random.seed(random_seed)

        # Get unique stations
        stations = np.array(dataset.target_stations)
        n_stations = len(stations)

        # Shuffle stations
        shuffled_indices = np.random.permutation(n_stations)
        shuffled_stations = stations[shuffled_indices]

        # Calculate splits
        n_val = int(n_stations * val_stations_ratio)
        n_test = int(n_stations * test_stations_ratio)
        n_train = n_stations - n_val - n_test

        train_stations = set(shuffled_stations[:n_train].tolist())
        val_stations = set(shuffled_stations[n_train:n_train + n_val].tolist())
        test_stations = set(shuffled_stations[n_train + n_val:].tolist())

        print(f"\nStation-based split:")
        print(f"  Train: {len(train_stations)} stations")
        print(f"  Val: {len(val_stations)} stations")
        print(f"  Test: {len(test_stations)} stations")

        # Find base sample indices for each split
        train_base_indices = [i for i, s in enumerate(dataset.sample_index)
                              if s['target_station'] in train_stations]
        val_base_indices = [i for i, s in enumerate(dataset.sample_index)
                            if s['target_station'] in val_stations]
        test_base_indices = [i for i, s in enumerate(dataset.sample_index)
                             if s['target_station'] in test_stations]

        print(f"  Train: {len(train_base_indices)} base samples -> {len(train_base_indices) * dataset.total_augmentations:,} augmented")
        print(f"  Val: {len(val_base_indices)} base samples -> {len(val_base_indices) * dataset.total_augmentations:,} augmented")
        print(f"  Test: {len(test_base_indices)} base samples -> {len(test_base_indices) * dataset.total_augmentations:,} augmented")

        # Create split datasets
        train_dataset = dataset._create_split(train_base_indices, list(train_stations))

        val_dataset = None
        if len(val_stations) > 0:
            val_dataset = dataset._create_split(val_base_indices, list(val_stations))

        test_dataset = None
        if len(test_stations) > 0:
            test_dataset = dataset._create_split(test_base_indices, list(test_stations))

        return train_dataset, val_dataset, test_dataset

    def _create_split(
        self,
        base_indices: List[int],
        stations: List[int]
    ) -> 'AugmentedLiveDataset':
        """Create a split dataset that references the same base data"""
        split = AugmentedLiveDataset.__new__(AugmentedLiveDataset)

        # Copy configuration
        split.seq_length = self.seq_length
        split.n_nearby_available = self.n_nearby_available
        split.n_nearby_in_features = self.n_nearby_in_features
        split.feature_params = self.feature_params
        split.is_prenormalized = self.is_prenormalized

        # Share base data (no copy!)
        split.base_features = self.base_features
        split.base_targets = self.base_targets
        split.base_masks = self.base_masks
        split.base_target_stations = self.base_target_stations
        split.base_end_dates = self.base_end_dates
        split.base_start_dates = self.base_start_dates
        split.n_base_samples = self.n_base_samples

        # Share base dataset reference if using that approach
        split._base_dataset = getattr(self, '_base_dataset', None)

        # For compatibility with code that checks for precomputed_data
        split.precomputed_data = getattr(self, 'precomputed_data', None)

        # Share normalization stats
        split.normalize = getattr(self, 'normalize', False)
        split.feature_mins = getattr(self, 'feature_mins', None)
        split.feature_maxs = getattr(self, 'feature_maxs', None)
        split.target_min = getattr(self, 'target_min', None)
        split.target_max = getattr(self, 'target_max', None)
        split.normalized_invalid_marker = getattr(self, 'normalized_invalid_marker', -2.0)

        # Share augmentation patterns
        split.skip_patterns = self.skip_patterns
        split.all_permutations = self.all_permutations
        split.total_augmentations = self.total_augmentations
        split._aug_column_indices = self._aug_column_indices

        # Share dimension info
        split.n_target_features = self.n_target_features
        split.nearby_features_per_station = self.nearby_features_per_station
        split.n_output_features = self.n_output_features
        split.target_cols = self.target_cols

        # Set split-specific data
        split.target_stations = stations
        split.sample_index = [self.sample_index[i] for i in base_indices]

        # Create index mapping for this split
        # Each base sample expands to total_augmentations samples
        split._base_indices = np.array(base_indices, dtype=np.int64)
        split._indices = None  # Not used in new logic

        return split

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        feature_names = []

        # Target station features
        for param in self.feature_params:
            feature_names.append(f'target_{param}')

        # Nearby stations features (output layout)
        for n_idx in range(self.n_nearby_in_features):
            feature_names.append(f'nearby{n_idx + 1}_distance')
            for param in self.feature_params:
                feature_names.append(f'nearby{n_idx + 1}_{param}')
            feature_names.append(f'nearby{n_idx + 1}_soil_moisture')

        return feature_names


class AugmentedPrecomputedDataset(Dataset):
    """
    Wrapper for the fully precomputed augmented dataset.
    
    This loads the precomputed augmented dataset (e.g., 120x augmented)
    and provides the same interface as AugmentedLiveDataset.
    
    Use this if you have disk space and want maximum speed.
    Use AugmentedLiveDataset if you want to save disk space.
    """

    def __init__(
        self,
        precomputed_path: str,
        norm_stats_path: Optional[str] = None,
    ):
        super().__init__()

        precomputed_path = Path(precomputed_path)
        if not precomputed_path.exists():
            raise ValueError(f"Precomputed path does not exist: {precomputed_path}")

        print(f"Loading precomputed augmented dataset from {precomputed_path}...")

        # Memory-map arrays
        self.features = np.load(precomputed_path / "features.npy", mmap_mode='r')
        self.targets = np.load(precomputed_path / "targets.npy", mmap_mode='r')
        self.masks = np.load(precomputed_path / "masks.npy", mmap_mode='r')

        # Load metadata
        self.target_stations_arr = np.load(precomputed_path / "target_stations.npy")
        self.end_dates = np.load(precomputed_path / "end_dates.npy")
        self.start_dates = np.load(precomputed_path / "start_dates.npy")

        # Check normalization
        if (precomputed_path / "is_normalized.npy").exists():
            self.is_prenormalized = bool(np.load(precomputed_path / "is_normalized.npy")[0])
        else:
            self.is_prenormalized = False

        # Build sample_index for split compatibility
        self.sample_index = []
        for i in range(len(self.features)):
            self.sample_index.append({
                'target_station': int(self.target_stations_arr[i]),
                'end_date': self.end_dates[i],
                'start_date': self.start_dates[i]
            })

        self.target_stations = list(np.unique(self.target_stations_arr))
        self._indices = None

        print(f"  Loaded {len(self.features):,} samples")
        print(f"  Shape: {self.features.shape}")
        print(f"  Pre-normalized: {self.is_prenormalized}")

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._indices is not None:
            idx = self._indices[idx]

        return {
            'features': torch.from_numpy(self.features[idx].copy()),
            'target': torch.from_numpy(self.targets[idx].copy()),
            'mask': torch.from_numpy(self.masks[idx].copy()),
            'target_station_id': int(self.target_stations_arr[idx]),
            'end_date': float(self.end_dates[idx])
        }

    @staticmethod
    def train_val_test_split(
        dataset: 'AugmentedPrecomputedDataset',
        val_stations_ratio: float = 0.15,
        test_stations_ratio: float = 0.0,
        random_seed: int = 42
    ) -> Tuple['AugmentedPrecomputedDataset', Optional['AugmentedPrecomputedDataset'], Optional['AugmentedPrecomputedDataset']]:
        """Split by stations"""
        np.random.seed(random_seed)

        stations = np.array(dataset.target_stations)
        n_stations = len(stations)

        shuffled_indices = np.random.permutation(n_stations)
        shuffled_stations = stations[shuffled_indices]

        n_val = int(n_stations * val_stations_ratio)
        n_test = int(n_stations * test_stations_ratio)
        n_train = n_stations - n_val - n_test

        train_stations = set(shuffled_stations[:n_train].tolist())
        val_stations = set(shuffled_stations[n_train:n_train + n_val].tolist())
        test_stations = set(shuffled_stations[n_train + n_val:].tolist())

        print(f"\nStation-based split:")
        print(f"  Train: {len(train_stations)} stations")
        print(f"  Val: {len(val_stations)} stations")
        print(f"  Test: {len(test_stations)} stations")

        # Get sample indices for each split
        train_indices = [i for i, s in enumerate(dataset.sample_index)
                         if s['target_station'] in train_stations]
        val_indices = [i for i, s in enumerate(dataset.sample_index)
                       if s['target_station'] in val_stations]
        test_indices = [i for i, s in enumerate(dataset.sample_index)
                        if s['target_station'] in test_stations]

        print(f"  Train: {len(train_indices):,} samples")
        print(f"  Val: {len(val_indices):,} samples")
        print(f"  Test: {len(test_indices):,} samples")

        train_dataset = dataset._create_split(train_indices, list(train_stations))
        val_dataset = dataset._create_split(val_indices, list(val_stations)) if val_indices else None
        test_dataset = dataset._create_split(test_indices, list(test_stations)) if test_indices else None

        return train_dataset, val_dataset, test_dataset

    def _create_split(
        self,
        indices: List[int],
        stations: List[int]
    ) -> 'AugmentedPrecomputedDataset':
        """Create a split that references the same arrays"""
        split = AugmentedPrecomputedDataset.__new__(AugmentedPrecomputedDataset)

        # Share data arrays (no copy!)
        split.features = self.features
        split.targets = self.targets
        split.masks = self.masks
        split.target_stations_arr = self.target_stations_arr
        split.end_dates = self.end_dates
        split.start_dates = self.start_dates
        split.is_prenormalized = self.is_prenormalized

        # Set split-specific data
        split.target_stations = stations
        split.sample_index = [self.sample_index[i] for i in indices]
        split._indices = np.array(indices, dtype=np.int64)

        return split


def benchmark_dataset(dataset, n_samples: int = 1000, batch_size: int = 64):
    """Benchmark dataset throughput"""
    import time
    from torch.utils.data import DataLoader

    print(f"\nBenchmarking {type(dataset).__name__}...")
    print(f"  Total samples: {len(dataset):,}")

    # Single sample access
    start = time.perf_counter()
    for i in range(n_samples):
        _ = dataset[i]
    single_time = time.perf_counter() - start
    print(f"  Single access: {n_samples / single_time:.0f} samples/sec")

    # DataLoader throughput
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    start = time.perf_counter()
    n_batches = 0
    n_total = 0
    for batch in loader:
        n_batches += 1
        n_total += len(batch['features'])
        if n_total >= n_samples:
            break
    loader_time = time.perf_counter() - start
    print(f"  DataLoader (4 workers): {n_total / loader_time:.0f} samples/sec")

    return single_time, loader_time


if __name__ == "__main__":
    """Example usage and benchmark"""
    from pathlib import Path
    import time
    import cProfile
    import pstats
    from io import StringIO

    data_dir = Path("meteogalicia_data")

    # Check what's available
    dense_path = data_dir / "dense_features.npz"
    timeseries_path = data_dir / "raw_timeseries.csv"
    stations_path = data_dir / "stations_metadata.csv"
    nearest_path = data_dir / "nearest_stations.csv"

    # Try to get filtered params from collector
    try:
        from Moisturizer import MeteoGaliciaCollector
        collector = MeteoGaliciaCollector(data_dir=str(data_dir))
        _, filtered_params = collector.analyze_parameter_coverage(coverage_threshold=0.25)
    except Exception as e:
        print(f"Could not load params from collector: {e}")
        filtered_params = None

    if dense_path.exists() and timeseries_path.exists():
        print("\n" + "=" * 70)
        print("Testing AugmentedLiveDataset (from dense arrays)")
        print("=" * 70)

        dataset = AugmentedLiveDataset.from_base_dataset(
            timeseries=str(timeseries_path),
            stations=str(stations_path),
            nearest=str(nearest_path),
            dense_array_path=str(dense_path),
            feature_params=filtered_params,
            seq_length=64,
            n_nearby_available=5,
            n_nearby_in_features=4,
            normalize=True,  # Will compute stats automatically
        )

        print(f"\nTotal samples: {len(dataset):,}")

        # ============================================================
        # DETAILED PROFILING
        # ============================================================
        print("\n" + "=" * 70)
        print("PROFILING __getitem__")
        print("=" * 70)

        # Profile individual components
        n_profile = 10000

        # 1. Time the index calculation
        print(f"\n1. Profiling index calculation ({n_profile} calls)...")
        start = time.perf_counter()
        for i in range(n_profile):
            base_idx, aug_idx = dataset._get_base_and_aug_idx(i)
        idx_time = time.perf_counter() - start
        print(f"   Index calc: {n_profile/idx_time:.0f} ops/sec ({idx_time*1000/n_profile:.4f} ms/op)")

        # 2. Time column index lookup
        print(f"\n2. Profiling column index lookup ({n_profile} calls)...")
        start = time.perf_counter()
        for i in range(n_profile):
            _, aug_idx = dataset._get_base_and_aug_idx(i)
            col_indices = dataset._aug_column_indices[aug_idx]
        col_time = time.perf_counter() - start
        print(f"   Column lookup: {n_profile/col_time:.0f} ops/sec ({col_time*1000/n_profile:.4f} ms/op)")

        # 3. Time base dataset fetch (the expensive part?)
        print(f"\n3. Profiling base dataset fetch ({n_profile} calls)...")
        start = time.perf_counter()
        for i in range(n_profile):
            base_idx, _ = dataset._get_base_and_aug_idx(i)
            sample = dataset._base_dataset[base_idx]
        fetch_time = time.perf_counter() - start
        print(f"   Base fetch: {n_profile/fetch_time:.0f} ops/sec ({fetch_time*1000/n_profile:.4f} ms/op)")

        # 4. Time column slicing
        print(f"\n4. Profiling column slicing ({n_profile} calls)...")
        sample = dataset._base_dataset[0]
        features_full = sample['features'].numpy()
        col_indices = dataset._aug_column_indices[0]
        start = time.perf_counter()
        for i in range(n_profile):
            features = features_full[:, col_indices].copy()
        slice_time = time.perf_counter() - start
        print(f"   Column slice: {n_profile/slice_time:.0f} ops/sec ({slice_time*1000/n_profile:.4f} ms/op)")

        # 5. Time tensor conversion
        print(f"\n5. Profiling tensor conversion ({n_profile} calls)...")
        features_np = features_full[:, col_indices].copy()
        start = time.perf_counter()
        for i in range(n_profile):
            t = torch.from_numpy(features_np)
        tensor_time = time.perf_counter() - start
        print(f"   Tensor conv: {n_profile/tensor_time:.0f} ops/sec ({tensor_time*1000/n_profile:.4f} ms/op)")

        # 6. Full __getitem__ timing
        print(f"\n6. Profiling full __getitem__ ({n_profile} calls)...")
        start = time.perf_counter()
        for i in range(n_profile):
            _ = dataset[i]
        full_time = time.perf_counter() - start
        print(f"   Full getitem: {n_profile/full_time:.0f} ops/sec ({full_time*1000/n_profile:.4f} ms/op)")

        # 7. Compare with v1's approach (if base dataset uses similar method)
        print(f"\n7. Direct base dataset access ({n_profile} calls)...")
        start = time.perf_counter()
        for i in range(n_profile):
            _ = dataset._base_dataset[i % len(dataset._base_dataset)]
        base_direct_time = time.perf_counter() - start
        print(f"   Base direct: {n_profile/base_direct_time:.0f} ops/sec ({base_direct_time*1000/n_profile:.4f} ms/op)")

        # Summary
        print("\n" + "=" * 70)
        print("TIMING BREAKDOWN")
        print("=" * 70)
        total_parts = idx_time + (col_time - idx_time) + (fetch_time - col_time) + slice_time + tensor_time
        print(f"  Index calc:     {idx_time/full_time*100:5.1f}%")
        print(f"  Base fetch:     {fetch_time/full_time*100:5.1f}%")
        print(f"  Column slice:   {slice_time/full_time*100:5.1f}%")
        print(f"  Tensor conv:    {tensor_time/full_time*100:5.1f}%")
        print(f"  Other overhead: {(full_time - fetch_time - slice_time - tensor_time)/full_time*100:5.1f}%")

        # cProfile for detailed breakdown
        print("\n" + "=" * 70)
        print("cProfile DETAILED BREAKDOWN (1000 calls)")
        print("=" * 70)
        
        pr = cProfile.Profile()
        pr.enable()
        for i in range(1000):
            _ = dataset[i]
        pr.disable()
        
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print(s.getvalue())

        # ============================================================
        # DataLoader test
        # ============================================================
        print("\n" + "=" * 70)
        print("DataLoader BENCHMARK")
        print("=" * 70)
        
        from torch.utils.data import DataLoader
        
        for num_workers in [0, 1, 2, 4, 8]:
            loader = DataLoader(
                dataset,
                batch_size=512,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            )
            
            # Warm up
            for batch in loader:
                break
            
            start = time.perf_counter()
            n_samples = 0
            for batch in loader:
                n_samples += len(batch['features'])
                if n_samples >= 20000:
                    break
            elapsed = time.perf_counter() - start
            print(f"  {num_workers} workers: {n_samples/elapsed:.0f} samples/sec")

    else:
        print(f"Required files not found in {data_dir}")
