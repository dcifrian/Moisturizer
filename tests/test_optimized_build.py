#!/usr/bin/env python3
"""
Test the optimized buildDataset() with a small dataset
"""

from Moisturizer import buildDataset

if __name__ == "__main__":
    print("Testing optimized buildDataset() with 16 days of data...")
    print("This should be very quick!")
    print("=" * 70)

    # Build a tiny dataset for testing
    train_ds, val_ds, test_ds = buildDataset(seq_length=8, days=16)

    print("\n" + "=" * 70)
    print("✓ TEST COMPLETE!")
    print("=" * 70)
    print(f"Train dataset size: {len(train_ds) if train_ds else 0}")
    print(f"Val dataset size: {len(val_ds) if val_ds else 0}")
    print(f"Test dataset size: {len(test_ds) if test_ds else 0}")

    # Try to get a sample
    if train_ds and len(train_ds) > 0:
        print("\nTrying to get a sample from train dataset...")
        sample = train_ds[0]
        print(f"✓ Sample retrieved successfully!")
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Target: {sample['target'].item():.4f}")
        print(f"  Mask shape: {sample['mask'].shape}")
