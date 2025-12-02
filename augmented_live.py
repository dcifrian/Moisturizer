#!/usr/bin/env python3
"""
Live Augmented Dataset - On-the-fly data augmentation without disk storage

Generates augmented samples dynamically by:
1. Loading base dataset with 5 nearby stations
2. Creating skip patterns (drop 1 of 5 stations)
3. Permuting the remaining 4 stations
4. Normalizing features using base dataset stats

This trades ~5 minutes per epoch for zero disk space and flexibility.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import permutations
from pathlib import Path
from typing import Optional, List
from Moisturizer import SoilMoistureSequenceDataset


class AugmentedLiveDataset(Dataset):
    """
    PyTorch Dataset that performs live augmentation without precomputing to disk.

    Args:
        timeseries: Path to timeseries CSV
        stations: Path to stations CSV
        nearest: Path to nearest neighbors CSV
        seq_length: Sequence length (default 64)
        n_nearby_available: Number of nearby stations in base dataset (default 5)
        n_nearby_in_features: Number of nearby stations in augmented samples (default 4)
        feature_params: List of feature parameters to include
        dense_array_path: Path to dense_features.npz (required for performance)

    The dataset will have len(base_dataset) × total_augmentations samples.
    Each base sample generates multiple augmented versions on-the-fly.
    """

    def __init__(
        self,
        timeseries: str,
        stations: str,
        nearest: str,
        seq_length: int = 64,
        n_nearby_available: int = 5,
        n_nearby_in_features: int = 4,
        feature_params: Optional[List[str]] = None,
        dense_array_path: Optional[str] = None,
    ):
        super().__init__()

        if dense_array_path is None or not Path(dense_array_path).exists():
            raise ValueError(
                "dense_array_path is required for AugmentedLiveDataset!\n"
                "Live augmentation needs dense arrays for fast performance.\n"
                f"Expected: {dense_array_path}"
            )

        self.seq_length = seq_length
        self.n_nearby_available = n_nearby_available
        self.n_nearby_in_features = n_nearby_in_features

        print("="*70)
        print("AUGMENTED LIVE DATASET - On-the-fly augmentation")
        print("="*70)

        # Load base dataset with n_nearby_available stations
        print(f"\n1. Loading base dataset with {n_nearby_available} nearby stations...")
        self.base_dataset = SoilMoistureSequenceDataset(
            timeseries=timeseries,
            stations=stations,
            nearest=nearest,
            seq_length=seq_length,
            n_nearest=n_nearby_available,
            feature_params=feature_params,
            precomputed_path=None,
            dense_array_path=dense_array_path,
            normalize=False  # We'll normalize during augmentation
        )

        print(f"   Base dataset: {len(self.base_dataset)} samples")

        # Generate augmentation patterns
        print(f"\n2. Generating augmentation patterns...")
        available_indices = list(range(n_nearby_available))

        # Skip patterns: which nearby stations to keep (drop 1 of 5)
        self.skip_patterns = []
        for skip_idx in range(n_nearby_available):
            keep_indices = [i for i in available_indices if i != skip_idx][:n_nearby_in_features]
            self.skip_patterns.append(keep_indices)

        # All permutations of the kept stations
        self.all_permutations = list(permutations(range(n_nearby_in_features)))

        self.total_augmentations = len(self.skip_patterns) * len(self.all_permutations)

        print(f"   Skip patterns: {len(self.skip_patterns)} (drop 1 of {n_nearby_available})")
        print(f"   Permutations: {len(self.all_permutations)} (of {n_nearby_in_features} stations)")
        print(f"   Total augmentations per sample: {self.total_augmentations}")
        print(f"   Total dataset size: {len(self.base_dataset) * self.total_augmentations:,} samples")

        # Calculate feature dimensions
        self.target_features_count = len(self.base_dataset.feature_params)
        self.nearby_features_per_station = 1 + len(self.base_dataset.feature_params) + 1  # distance + features + soil
        self.total_features = self.target_features_count + (self.nearby_features_per_station * n_nearby_in_features)

        print(f"\n3. Feature dimensions:")
        print(f"   Target features: {self.target_features_count}")
        print(f"   Nearby features per station: {self.nearby_features_per_station}")
        print(f"   Total features: {self.total_features}")

        # Load normalization stats from base dataset
        print(f"\n4. Computing normalization stats from base dataset...")
        self._compute_normalization_stats()

        print("\n✓ Augmented live dataset ready!")
        print("="*70)

    def _compute_normalization_stats(self):
        """
        Compute normalization stats from base dataset samples.
        Uses the same method as precompute_augmented.py --use-base-stats
        """
        num_samples_for_stats = min(10000, len(self.base_dataset))
        sample_indices = np.random.choice(
            len(self.base_dataset),
            size=num_samples_for_stats,
            replace=False
        )

        print(f"   Sampling {num_samples_for_stats} samples from base dataset...")

        # Initialize min/max tracking
        feature_mins = np.full(self.total_features, np.inf, dtype=np.float32)
        feature_maxs = np.full(self.total_features, -np.inf, dtype=np.float32)
        target_min = np.inf
        target_max = -np.inf

        invalid_markers = [-9999.0, -1000.0]

        for idx in sample_indices:
            sample = self.base_dataset[int(idx)]
            features = sample['features'].numpy()
            target = sample['target'].numpy()[0]

            # Target stats
            if target not in invalid_markers:
                target_min = min(target_min, target)
                target_max = max(target_max, target)

            # Target station features
            target_feats = features[:, :self.target_features_count]
            for feat_idx in range(self.target_features_count):
                feat_values = target_feats[:, feat_idx]
                valid = feat_values[(feat_values != -1000.0) & (feat_values != -9999.0)]
                if len(valid) > 0:
                    feature_mins[feat_idx] = min(feature_mins[feat_idx], valid.min())
                    feature_maxs[feat_idx] = max(feature_maxs[feat_idx], valid.max())

            # Nearby stations: Extract all 5 stations' data
            nearby_start = self.target_features_count
            nearby_base = features[:, nearby_start:].reshape(
                self.seq_length, self.n_nearby_available, self.nearby_features_per_station
            )

            # For each feature across nearby stations
            for nearby_feat_idx in range(self.nearby_features_per_station):
                feat_across_stations = nearby_base[:, :, nearby_feat_idx]
                valid = feat_across_stations[
                    (feat_across_stations != -1000.0) & (feat_across_stations != -9999.0)
                ]

                if len(valid) > 0:
                    # Apply same range to all 4 slots in augmented dataset
                    for slot in range(self.n_nearby_in_features):
                        aug_feat_idx = (
                            self.target_features_count +
                            (slot * self.nearby_features_per_station) +
                            nearby_feat_idx
                        )
                        feature_mins[aug_feat_idx] = min(feature_mins[aug_feat_idx], valid.min())
                        feature_maxs[aug_feat_idx] = max(feature_maxs[aug_feat_idx], valid.max())

        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.normalized_invalid_marker = -2.0

        print(f"   Feature range: [{feature_mins.min():.2f}, {feature_maxs.max():.2f}]")
        print(f"   Target range: [{target_min:.2f}, {target_max:.2f}]")

    def __len__(self):
        """Total number of augmented samples"""
        return len(self.base_dataset) * self.total_augmentations

    def __getitem__(self, idx: int):
        """
        Get an augmented sample by:
        1. Mapping augmented idx -> base_idx + augmentation pattern
        2. Fetching base sample
        3. Applying augmentation (skip + permute)
        4. Normalizing features
        """
        # Map augmented index to base index and augmentation index
        base_idx = idx // self.total_augmentations
        aug_idx = idx % self.total_augmentations

        # Determine which skip pattern and permutation to use
        skip_idx = aug_idx // len(self.all_permutations)
        perm_idx = aug_idx % len(self.all_permutations)

        # Get base sample from dataset
        base_sample = self.base_dataset[base_idx]
        base_features = base_sample['features'].numpy()
        base_mask = base_sample['mask'].numpy()
        base_target = base_sample['target'].numpy()[0]

        # Apply augmentation
        aug_features, aug_mask = self._apply_augmentation(
            base_features, base_mask, skip_idx, perm_idx
        )

        # Normalize
        normalized_features, normalized_target = self._normalize(
            aug_features, aug_mask, base_target
        )

        return {
            'features': torch.from_numpy(normalized_features),
            'target': torch.tensor([normalized_target], dtype=torch.float32),
            'mask': torch.from_numpy(aug_mask),
            'target_station_id': base_sample['target_station_id'],
            'end_date': base_sample['end_date']
        }

    def _apply_augmentation(
        self,
        base_features: np.ndarray,
        base_mask: np.ndarray,
        skip_idx: int,
        perm_idx: int
    ):
        """
        Apply augmentation: skip pattern + permutation

        Args:
            base_features: [seq_length, total_features_base]
            base_mask: [seq_length, total_features_base]
            skip_idx: Which skip pattern to use
            perm_idx: Which permutation to use

        Returns:
            aug_features: [seq_length, total_features_augmented]
            aug_mask: [seq_length, total_features_augmented]
        """
        # Extract target station features (unchanged)
        target_feat = base_features[:, :self.target_features_count]
        target_mask = base_mask[:, :self.target_features_count]

        # Extract nearby stations and reshape
        nearby_start = self.target_features_count
        nearby_features = base_features[:, nearby_start:].reshape(
            self.seq_length, self.n_nearby_available, self.nearby_features_per_station
        )
        nearby_mask = base_mask[:, nearby_start:].reshape(
            self.seq_length, self.n_nearby_available, self.nearby_features_per_station
        )

        # Apply skip pattern (select 4 of 5 stations)
        keep_indices = self.skip_patterns[skip_idx]
        nearby_features_4 = nearby_features[:, keep_indices, :]
        nearby_mask_4 = nearby_mask[:, keep_indices, :]

        # Apply permutation
        perm = self.all_permutations[perm_idx]
        perm_nearby_features = nearby_features_4[:, perm, :].reshape(self.seq_length, -1)
        perm_nearby_mask = nearby_mask_4[:, perm, :].reshape(self.seq_length, -1)

        # Concatenate target + permuted nearby
        aug_features = np.concatenate([target_feat, perm_nearby_features], axis=1)
        aug_mask = np.concatenate([target_mask, perm_nearby_mask], axis=1)

        return aug_features, aug_mask

    def _normalize(self, features: np.ndarray, mask: np.ndarray, target: float):
        """
        Normalize features and target using precomputed stats

        Args:
            features: [seq_length, total_features]
            mask: [seq_length, total_features]
            target: Scalar target value

        Returns:
            normalized_features: [seq_length, total_features]
            normalized_target: Scalar
        """
        invalid_markers = [-9999.0, -1000.0]

        # Normalize features (vectorized)
        normalized_features = features.copy()
        invalid_mask = np.isin(normalized_features, invalid_markers)

        feat_ranges = self.feature_maxs - self.feature_mins
        valid_ranges = feat_ranges > 0

        # Normalize all features at once
        normalized_features = 2.0 * (
            (normalized_features - self.feature_mins[None, :]) /
            np.where(valid_ranges[None, :], feat_ranges[None, :], 1.0)
        ) - 1.0

        # Set invalid values
        normalized_features[invalid_mask] = self.normalized_invalid_marker

        # Normalize target
        if target in invalid_markers:
            normalized_target = self.normalized_invalid_marker
        else:
            if self.target_max > self.target_min:
                normalized_target = 2.0 * (target - self.target_min) / (self.target_max - self.target_min) - 1.0
            else:
                normalized_target = target

        return normalized_features, normalized_target


if __name__ == "__main__":
    """
    Example usage and simple test
    """
    from pathlib import Path

    # Example paths (adjust to your setup)
    data_dir = Path("meteogalicia_data")

    dataset = AugmentedLiveDataset(
        timeseries=str(data_dir / "raw_timeseries.csv"),
        stations=str(data_dir / "stations.csv"),
        nearest=str(data_dir / "nearest.csv"),
        seq_length=64,
        n_nearby_available=5,
        n_nearby_in_features=4,
        dense_array_path=str(data_dir / "dense_features.npz")
    )

    print(f"\nTesting dataset access...")
    print(f"Total samples: {len(dataset):,}")

    # Test a few samples
    for i in [0, 100, 1000]:
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Target: {sample['target'].item():.4f}")
        print(f"  Mask shape: {sample['mask'].shape}")
