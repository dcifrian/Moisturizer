#!/usr/bin/env python3
"""Test buildDataset with real data - small dataset"""

from Moisturizer import buildDataset

print("="*70)
print("TESTING buildDataset WITH REAL DATA")
print("Parameters: seq_length=4, days=16")
print("="*70)

try:
    train_ds, val_ds, test_ds = buildDataset(seq_length=4, days=16)

    print("\n" + "="*70)
    print("BUILD SUCCESSFUL!")
    print("="*70)

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)}")
    print(f"  Val: {len(val_ds) if val_ds else 0}")
    print(f"  Test: {len(test_ds) if test_ds else 0}")

    if len(train_ds) > 0:
        print("\nGetting first sample...")
        sample = train_ds[0]
        print(f"  Features shape: {sample['features'].shape}")
        print(f"  Target: {sample['target']}")
        print(f"  Target value: {sample['target'].item()}")
        print(f"  Mask shape: {sample['mask'].shape}")
        print(f"  Station ID: {sample['target_station_id']}")

        target_val = sample['target'].item()
        if target_val == -1000.0:
            print(f"\n✗ ERROR: Target is -1000.0 (invalid marker)!")
        elif target_val < 0.0 or target_val > 1.0:
            print(f"\n✗ ERROR: Target {target_val} is out of range [0.0, 1.0]!")
        else:
            print(f"\n✓ Target is valid: {target_val:.4f}")

except Exception as e:
    print("\n" + "="*70)
    print("BUILD FAILED WITH ERROR:")
    print("="*70)
    import traceback
    traceback.print_exc()
