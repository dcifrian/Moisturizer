#!/usr/bin/env python3
"""
Torch.compile Optimized Augmented Live Dataset

Uses PyTorch operations instead of numpy and torch.compile to fuse operations
into optimized kernels. This eliminates Python overhead and allows the compiler
to optimize the entire augmentation + normalization pipeline.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import permutations
from pathlib import Path
from typing import Optional, List
from Moisturizer import SoilMoistureSequenceDataset


class AugmentedLiveDatasetTorch(Dataset):
    """
    PyTorch Dataset with torch.compile optimized augmentation.

    Uses pure torch operations for augmentation and normalization,
    allowing torch.compile to fuse everything into optimized kernels.
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
                "dense_array_path is required for AugmentedLiveDatasetTorch!\n"
                f"Expected: {dense_array_path}"
            )

        self.seq_length = seq_length
        self.n_nearby_available = n_nearby_available
        self.n_nearby_in_features = n_nearby_in_features

        print("="*70)
        print("TORCH.COMPILE OPTIMIZED AUGMENTED LIVE DATASET")
        print("="*70)

        # Load base dataset
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
            normalize=False
        )

        print(f"   Base dataset: {len(self.base_dataset)} samples")

        # Generate augmentation patterns as torch tensors
        print(f"\n2. Generating augmentation patterns...")
        available_indices = list(range(n_nearby_available))

        # Skip patterns: which nearby stations to keep (drop 1 of 5)
        skip_patterns_list = []
        for skip_idx in range(n_nearby_available):
            keep_indices = [i for i in available_indices if i != skip_idx][:n_nearby_in_features]
            skip_patterns_list.append(keep_indices)

        # Convert to torch tensor for GPU-friendly indexing
        self.skip_patterns = torch.tensor(skip_patterns_list, dtype=torch.long)

        # All permutations as torch tensor
        all_perms = list(permutations(range(n_nearby_in_features)))
        self.all_permutations = torch.tensor(all_perms, dtype=torch.long)

        self.total_augmentations = len(skip_patterns_list) * len(all_perms)

        print(f"   Skip patterns: {len(skip_patterns_list)} (drop 1 of {n_nearby_available})")
        print(f"   Permutations: {len(all_perms)} (of {n_nearby_in_features} stations)")
        print(f"   Total augmentations per sample: {self.total_augmentations}")
        print(f"   Total dataset size: {len(self.base_dataset) * self.total_augmentations:,} samples")

        # Feature dimensions
        self.target_features_count = len(self.base_dataset.feature_params)
        self.nearby_features_per_station = 1 + len(self.base_dataset.feature_params) + 1
        self.total_features = self.target_features_count + (self.nearby_features_per_station * n_nearby_in_features)

        print(f"\n3. Feature dimensions:")
        print(f"   Target features: {self.target_features_count}")
        print(f"   Nearby features per station: {self.nearby_features_per_station}")
        print(f"   Total features: {self.total_features}")

        # Compute normalization stats
        print(f"\n4. Computing normalization stats from base dataset...")
        self._compute_normalization_stats()

        # Create compiled augmentation function
        print(f"\n5. Creating torch.compile optimized function...")
        self._augment_and_normalize_compiled = torch.compile(
            self._augment_and_normalize,
            mode='reduce-overhead',  # Optimize for low latency
            fullgraph=True  # Compile entire function as one graph
        )
        print(f"   ✓ Compiled with mode='reduce-overhead'")

        print("\n✓ Torch.compile augmented dataset ready!")
        print("="*70)

    def _compute_normalization_stats(self):
        """Compute normalization stats from base dataset samples"""
        num_samples_for_stats = min(10000, len(self.base_dataset))
        sample_indices = np.random.choice(
            len(self.base_dataset),
            size=num_samples_for_stats,
            replace=False
        )

        print(f"   Sampling {num_samples_for_stats} samples from base dataset...")

        # Initialize min/max tracking
        feature_mins = torch.full((self.total_features,), float('inf'))
        feature_maxs = torch.full((self.total_features,), float('-inf'))
        target_min = float('inf')
        target_max = float('-inf')

        for idx in sample_indices:
            sample = self.base_dataset[int(idx)]
            features = sample['features']
            target = sample['target'][0].item()

            # Target stats
            if target not in [-9999.0, -1000.0]:
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

            # Nearby stations: all 5 stations' data
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

        # Store as torch tensors for compiled function
        self.feature_mins = feature_mins
        self.feature_maxs = feature_maxs
        self.target_min = torch.tensor(target_min, dtype=torch.float32)
        self.target_max = torch.tensor(target_max, dtype=torch.float32)

        print(f"   Feature range: [{feature_mins.min():.2f}, {feature_maxs.max():.2f}]")
        print(f"   Target range: [{target_min:.2f}, {target_max:.2f}]")

    def __len__(self):
        """Total number of augmented samples"""
        return len(self.base_dataset) * self.total_augmentations

    def __getitem__(self, idx: int):
        """Get an augmented sample using compiled torch operations"""
        # Map augmented index to base index and augmentation index
        base_idx = idx // self.total_augmentations
        aug_idx = idx % self.total_augmentations

        # Determine which skip pattern and permutation to use
        skip_idx = aug_idx // len(self.all_permutations)
        perm_idx = aug_idx % len(self.all_permutations)

        # Get base sample (already torch tensors!)
        base_sample = self.base_dataset[base_idx]
        base_features = base_sample['features']  # Keep as torch tensor
        base_mask = base_sample['mask']  # Keep as torch tensor
        base_target = base_sample['target'][0]  # Keep as torch scalar

        # Get augmentation indices
        skip_pattern = self.skip_patterns[skip_idx]
        perm_pattern = self.all_permutations[perm_idx]

        # Call compiled function
        aug_features, aug_target, aug_mask = self._augment_and_normalize_compiled(
            base_features,
            base_mask,
            base_target,
            skip_pattern,
            perm_pattern,
            self.feature_mins,
            self.feature_maxs,
            self.target_min,
            self.target_max,
            self.target_features_count,
            self.nearby_features_per_station,
            self.n_nearby_available,
            self.n_nearby_in_features,
            self.seq_length
        )

        return {
            'features': aug_features,
            'target': aug_target.unsqueeze(0),
            'mask': aug_mask,
            'target_station_id': base_sample['target_station_id'],
            'end_date': base_sample['end_date']
        }

    def _augment_and_normalize(
        self,
        base_features: torch.Tensor,
        base_mask: torch.Tensor,
        base_target: torch.Tensor,
        skip_pattern: torch.Tensor,
        perm_pattern: torch.Tensor,
        feature_mins: torch.Tensor,
        feature_maxs: torch.Tensor,
        target_min: torch.Tensor,
        target_max: torch.Tensor,
        target_features_count: int,
        nearby_features_per_station: int,
        n_nearby_available: int,
        n_nearby_in_features: int,
        seq_length: int
    ):
        """
        Augment and normalize in pure torch operations (compiled).

        This function will be compiled by torch.compile into optimized kernels.
        All operations are torch operations - no numpy, no Python loops.
        """
        # Extract target station features (unchanged)
        target_feat = base_features[:, :target_features_count]
        target_mask = base_mask[:, :target_features_count]

        # Extract and reshape nearby stations
        nearby_start = target_features_count
        nearby_features = base_features[:, nearby_start:].reshape(
            seq_length, n_nearby_available, nearby_features_per_station
        )
        nearby_mask = base_mask[:, nearby_start:].reshape(
            seq_length, n_nearby_available, nearby_features_per_station
        )

        # Apply skip pattern (select 4 of 5 stations)
        nearby_features_4 = nearby_features[:, skip_pattern, :]
        nearby_mask_4 = nearby_mask[:, skip_pattern, :]

        # Apply permutation
        perm_nearby_features = nearby_features_4[:, perm_pattern, :].reshape(seq_length, -1)
        perm_nearby_mask = nearby_mask_4[:, perm_pattern, :].reshape(seq_length, -1)

        # Concatenate target + permuted nearby
        aug_features = torch.cat([target_feat, perm_nearby_features], dim=1)
        aug_mask = torch.cat([target_mask, perm_nearby_mask], dim=1)

        # Normalize features - fully vectorized, no loops or conditionals
        # This is critical for torch.compile to work!

        # Compute ranges (avoid division by zero)
        ranges = feature_maxs - feature_mins
        ranges = torch.where(ranges > 0, ranges, torch.ones_like(ranges))

        # Create invalid mask
        invalid_mask = (aug_features == -1000.0) | (aug_features == -9999.0)

        # Normalize to [-1, 1]: 2 * (x - min) / (max - min) - 1
        normalized_features = 2.0 * (aug_features - feature_mins) / ranges - 1.0

        # Set invalid values to -2.0
        normalized_features = torch.where(invalid_mask, -2.0, normalized_features)

        # Set features with no range (max == min) to -2.0
        no_range_mask = (feature_maxs - feature_mins) <= 0
        normalized_features[:, no_range_mask] = -2.0

        # Normalize target - using torch.where to avoid conditionals
        target_invalid = (base_target == -9999.0) | (base_target == -1000.0)
        target_range = target_max - target_min
        target_has_range = target_range > 0

        # Compute normalized value
        normalized_target = 2.0 * (base_target - target_min) / torch.where(
            target_has_range, target_range, torch.ones_like(target_range)
        ) - 1.0

        # Use -2.0 for invalid targets or targets with no range
        normalized_target = torch.where(
            target_invalid | ~target_has_range,
            torch.tensor(-2.0, dtype=torch.float32),
            normalized_target
        )

        return normalized_features, normalized_target, aug_mask


if __name__ == "__main__":
    """Example usage"""
    from pathlib import Path

    data_dir = Path("meteogalicia_data")

    dense_data = np.load(str(data_dir / "dense_features.npz"))
    feature_params = [p for p in dense_data['feature_params'].tolist() if p != 'HS_CV_AVG_-0.2m']

    dataset = AugmentedLiveDatasetTorch(
        timeseries=str(data_dir / "raw_timeseries.csv"),
        stations=str(data_dir / "stations_metadata.csv"),
        nearest=str(data_dir / "nearest_stations.csv"),
        seq_length=2,
        n_nearby_available=5,
        n_nearby_in_features=4,
        feature_params=feature_params,
        dense_array_path=str(data_dir / "dense_features.npz")
    )

    print(f"\nTesting dataset access...")
    print(f"Total samples: {len(dataset):,}")

    # Test a few samples (this will trigger compilation on first call)
    for i in [0, 100, 1000]:
        if i >= len(dataset):
            continue
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Target: {sample['target'].item():.4f}")
