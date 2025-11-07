#!/usr/bin/env python3
"""
Example: Training with augmented dataset

Shows how to use AugmentedSoilMoistureDataset for improved spatial generalization
"""

import torch
from Moisturizer import MeteoGaliciaCollector, SoilMoistureSequenceDataset
from augmented_dataset import AugmentedSoilMoistureDataset

def main():
    print("=" * 70)
    print("TRAINING WITH SPATIAL AUGMENTATION")
    print("=" * 70)

    # 1. Load base dataset (your existing precomputed data)
    collector = MeteoGaliciaCollector()

    print("\n1. Loading base dataset...")
    base_dataset = SoilMoistureSequenceDataset(
        timeseries=str(collector.timeseries_file),
        stations=str(collector.stations_file),
        nearest=str(collector.nearest_file),
        seq_length=64,
        n_nearest=4,
        feature_params=None,  # Auto-determined from analyze_parameter_coverage
        precomputed_path=str(collector.data_dir / "precomputed_sequences.npz"),
        normalize=True,
        norm_stats_path=str(collector.data_dir / "normalization_stats.npz")
    )

    print(f"   Base dataset: {len(base_dataset)} samples")

    # 2. Split into train/val
    print("\n2. Splitting train/val...")
    train_dataset, val_dataset = base_dataset.train_val_test_split(
        train_fraction=0.8,
        val_fraction=0.2,
        test_fraction=0.0
    )

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # 3. Wrap train dataset with augmentation
    print("\n3. Creating augmented training dataset...")
    aug_train_dataset = AugmentedSoilMoistureDataset(
        base_dataset=train_dataset,
        augmentation_modes=['shuffle', 'shift', 'skip'],
        augmentation_prob=0.8,  # 80% of samples get augmented
        samples_per_base=4,     # Each base sample -> 4 augmented samples
        use_extended_stations=False  # Fast mode: permute existing 4 stations
    )

    # Val dataset: NO augmentation (evaluate on original data)
    print("\n4. Validation dataset (no augmentation)...")
    print(f"   Val: {len(val_dataset)} samples (original)")

    # 5. Create DataLoaders
    print("\n5. Creating DataLoaders...")

    train_loader = torch.utils.data.DataLoader(
        aug_train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=10
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")

    # 6. Test augmentation
    print("\n6. Testing augmentation...")
    base_sample = train_dataset[0]
    aug_sample1 = aug_train_dataset[0]
    aug_sample2 = aug_train_dataset[len(train_dataset)]  # Same base, different aug

    print(f"   Base sample features shape: {base_sample['features'].shape}")
    print(f"   Aug sample 1 features shape: {aug_sample1['features'].shape}")
    print(f"   Aug sample 2 features shape: {aug_sample2['features'].shape}")

    # Check if augmentation is working
    nearby_start = 26  # Where nearby stations start
    base_nearby = base_sample['features'][:, nearby_start:nearby_start+10]
    aug_nearby = aug_sample1['features'][:, nearby_start:nearby_start+10]

    features_changed = not torch.all(base_nearby == aug_nearby)
    print(f"   Nearby stations changed: {features_changed} (should be True)")

    # 7. Example training loop snippet
    print("\n7. Example training loop:")
    print("""
    for epoch in range(num_epochs):
        # Training with augmentation
        for batch in train_loader:
            features = batch['features']  # Different each epoch!
            targets = batch['target']

            # Your training code here...
            loss = model(features, targets)
            loss.backward()
            optimizer.step()

        # Validation without augmentation
        for batch in val_loader:
            features = batch['features']  # Same each epoch
            targets = batch['target']

            # Your validation code here...
            val_loss = model(features, targets)
    """)

    print("\n" + "=" * 70)
    print("✓ READY TO TRAIN!")
    print("=" * 70)
    print("\nBenefits of augmentation:")
    print("  ✓ Train dataset: 4x larger with spatial variations")
    print("  ✓ Model learns position-invariant features")
    print("  ✓ Better generalization to unseen station combinations")
    print("  ✓ Validation on original data (fair comparison)")
    print("\nExpected improvements:")
    print("  - Better predictions on moisture map (unseen locations)")
    print("  - More robust to missing nearby stations")
    print("  - Slightly slower training (4x samples) but better quality")
    print("=" * 70)

if __name__ == "__main__":
    main()
