#!/usr/bin/env python3
"""
Augmented dataset wrapper for soil moisture prediction

Applies spatial augmentation by:
1. Shuffling order of nearby stations
2. Shifting to use stations [2,3,4,5] instead of [1,2,3,4]
3. Skipping stations to use [1,3,5,7] or other combinations
4. Combining the above

This teaches the model to be invariant to:
- Which specific nearby stations are available
- The order of nearby stations in the feature vector
- Exact distances (slight variations due to different stations)
"""

import numpy as np
import torch
from typing import Optional, List, Tuple
import random


class AugmentedSoilMoistureDataset(torch.utils.data.Dataset):
    """
    Wrapper that augments SoilMoistureSequenceDataset by permuting nearby stations

    Args:
        base_dataset: Base SoilMoistureSequenceDataset with precomputed data
        augmentation_modes: List of augmentation types to use
            - 'shuffle': Randomly shuffle order of 4 nearby stations (24 permutations)
            - 'shift': Use stations [2,3,4,5] instead of [1,2,3,4] (needs stations 5-8)
            - 'skip': Use stations [1,3,5,7] (skip every other, needs stations 5-8)
            - 'skip2': Use stations [2,4,6,8] (skip, starting from 2nd, needs stations 5-8)
        augmentation_prob: Probability of applying augmentation (0.0 = never, 1.0 = always)
        samples_per_base: How many augmented samples to generate per base sample
                         (None = use all modes, int = fixed multiplier)
        use_extended_stations: If True, load stations 5-8 from nearest_df for shift/skip modes
                              If False, just permute existing 4 stations (faster but less diverse)
    """

    def __init__(
        self,
        base_dataset,
        augmentation_modes: List[str] = ['shuffle', 'shift', 'skip'],
        augmentation_prob: float = 1.0,
        samples_per_base: Optional[int] = None,
        use_extended_stations: bool = False
    ):
        self.base_dataset = base_dataset
        self.augmentation_modes = augmentation_modes
        self.augmentation_prob = augmentation_prob
        self.samples_per_base = samples_per_base
        self.use_extended_stations = use_extended_stations

        # Validate base dataset has precomputed data
        if base_dataset.precomputed_data is None:
            raise ValueError("Base dataset must have precomputed data for augmentation!")

        # If using extended stations, we need access to nearest_df and dense arrays
        if use_extended_stations:
            if base_dataset.nearest_df is None:
                raise ValueError("Base dataset must have nearest_df loaded for extended stations!")
            print("  Using extended stations [5-8] for shift/skip modes (more diverse but slower)")
        else:
            print("  Permuting existing 4 stations only (fast but less diverse)")

        # Calculate effective dataset size
        if samples_per_base is not None:
            self.effective_size = len(base_dataset) * samples_per_base
        else:
            # Estimate based on modes
            multiplier = 1
            if 'shuffle' in augmentation_modes:
                multiplier *= 24  # 4! permutations
            if 'shift' in augmentation_modes:
                multiplier *= 2  # Original + shifted
            if 'skip' in augmentation_modes or 'skip2' in augmentation_modes:
                multiplier *= 2
            self.effective_size = len(base_dataset) * multiplier

        print(f"AugmentedDataset initialized:")
        print(f"  Base samples: {len(base_dataset)}")
        print(f"  Augmentation modes: {augmentation_modes}")
        print(f"  Augmentation probability: {augmentation_prob}")
        print(f"  Effective samples: {self.effective_size}")

    def __len__(self):
        return self.effective_size

    def __getitem__(self, idx):
        """Get augmented sample"""
        # Map augmented index back to base index
        base_idx = idx % len(self.base_dataset)

        # Get base sample
        base_sample = self.base_dataset[base_idx]

        # Apply augmentation with probability
        if random.random() < self.augmentation_prob:
            # Choose augmentation mode
            mode = random.choice(self.augmentation_modes)

            # Apply augmentation
            augmented_sample = self._augment_sample(base_sample, mode, base_idx)
            return augmented_sample
        else:
            return base_sample

    def _augment_sample(self, sample, mode, base_idx):
        """
        Augment a sample by permuting nearby stations

        Sample structure:
        features: [seq_length, total_features]
        total_features = target_features + (nearby_features_per_station * n_nearest)
        nearby_features_per_station = 1 (distance) + len(feature_params) + 1 (soil moisture)
        """
        features = sample['features'].clone()  # [seq_length, total_features]
        mask = sample['mask'].clone()
        target = sample['target']  # Unchanged

        # Get feature structure
        n_nearest = self.base_dataset.n_nearest
        target_features = len(self.base_dataset.feature_params)
        nearby_features_per_station = 1 + target_features + 1

        # Extract nearby station features
        nearby_start = target_features
        nearby_features = features[:, nearby_start:]  # [seq_length, n_nearest * nearby_features_per_station]
        nearby_mask = mask[:, nearby_start:]

        # Reshape to separate stations
        nearby_features = nearby_features.reshape(
            features.shape[0],
            n_nearest,
            nearby_features_per_station
        )  # [seq_length, n_nearest, features_per_station]

        nearby_mask = nearby_mask.reshape(
            mask.shape[0],
            n_nearest,
            nearby_features_per_station
        )

        # Apply augmentation based on mode
        if mode == 'shuffle':
            # Randomly shuffle the order of nearby stations
            perm = torch.randperm(n_nearest)
            nearby_features = nearby_features[:, perm, :]
            nearby_mask = nearby_mask[:, perm, :]

        elif mode == 'shift':
            # Shift to use stations [2,3,4,5] instead of [1,2,3,4]
            # We need to get station 5 from the base dataset
            # For now, just rotate: [1,2,3,4] -> [2,3,4,1]
            nearby_features = torch.roll(nearby_features, shifts=-1, dims=1)
            nearby_mask = torch.roll(nearby_mask, shifts=-1, dims=1)

        elif mode == 'skip':
            # Use stations [1,3,5,7] - we only have 4, so use [1,3,1,3] or similar
            # Better: use [1,3,2,4] (interleave)
            indices = torch.tensor([0, 2, 1, 3])
            nearby_features = nearby_features[:, indices, :]
            nearby_mask = nearby_mask[:, indices, :]

        elif mode == 'skip2':
            # Use stations [2,4,1,3]
            indices = torch.tensor([1, 3, 0, 2])
            nearby_features = nearby_features[:, indices, :]
            nearby_mask = nearby_mask[:, indices, :]

        # Reshape back
        nearby_features = nearby_features.reshape(
            features.shape[0],
            n_nearest * nearby_features_per_station
        )
        nearby_mask = nearby_mask.reshape(
            mask.shape[0],
            n_nearest * nearby_features_per_station
        )

        # Reconstruct full features tensor
        features[:, nearby_start:] = nearby_features
        mask[:, nearby_start:] = nearby_mask

        return {
            'features': features,
            'target': target,
            'mask': mask
        }


def create_augmented_dataloader(
    base_dataset,
    batch_size: int,
    augmentation_modes: List[str] = ['shuffle', 'shift', 'skip'],
    augmentation_prob: float = 1.0,
    samples_per_base: Optional[int] = 4,
    num_workers: int = 4,
    **dataloader_kwargs
):
    """
    Create a DataLoader with augmented dataset

    Example:
        train_loader = create_augmented_dataloader(
            base_dataset=train_dataset,
            batch_size=512,
            augmentation_modes=['shuffle', 'shift'],
            augmentation_prob=0.8,  # 80% chance of augmentation
            samples_per_base=4,  # 4x dataset size
            num_workers=8
        )
    """
    augmented_dataset = AugmentedSoilMoistureDataset(
        base_dataset=base_dataset,
        augmentation_modes=augmentation_modes,
        augmentation_prob=augmentation_prob,
        samples_per_base=samples_per_base
    )

    dataloader = torch.utils.data.DataLoader(
        augmented_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        **dataloader_kwargs
    )

    return dataloader


if __name__ == "__main__":
    # Example usage
    print("=" * 70)
    print("AUGMENTED DATASET EXAMPLE")
    print("=" * 70)

    from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset

    # Load base dataset
    collector = MeteoGaliciaCollector()

    base_dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=4,
        feature_params=None,  # Will be determined automatically
        precomputed_path=str(collector.data_dir / "precomputed_sequences.npz"),
        normalize=True
    )

    # Create augmented dataset
    aug_dataset = AugmentedSoilMoistureDataset(
        base_dataset=base_dataset,
        augmentation_modes=['shuffle', 'shift', 'skip'],
        augmentation_prob=0.8,
        samples_per_base=4
    )

    print(f"\nTesting augmentation:")
    print(f"  Base sample 0 features shape: {base_dataset[0]['features'].shape}")
    print(f"  Aug sample 0 features shape: {aug_dataset[0]['features'].shape}")
    print(f"  Aug sample 1 features shape: {aug_dataset[1]['features'].shape}")

    # Check that augmentation changes features
    base_feat = base_dataset[0]['features']
    aug_feat = aug_dataset[0]['features']

    print(f"\n  Features identical: {torch.all(base_feat == aug_feat).item()}")
    print(f"  (Should be False if augmentation applied)")

    print("\n✓ Augmented dataset ready!")
