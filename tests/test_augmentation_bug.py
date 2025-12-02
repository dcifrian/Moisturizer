#!/usr/bin/env python3
"""
Debug script to trace the augmentation bug causing wrong target ranges.
"""
import numpy as np
from pathlib import Path

# Simulate what the worker does
def test_memmap_write():
    """Test if there's a dtype issue with memmap writes"""
    test_dir = Path("./test_memmap_debug")
    test_dir.mkdir(exist_ok=True)

    # Create targets array like we do in the main process
    print("Creating memmap file...")
    targets_path = str(test_dir / "targets.npy")
    all_targets_create = np.lib.format.open_memmap(
        targets_path, dtype=np.float32, mode='w+',
        shape=(100, 1)
    )
    print(f"Created: dtype={all_targets_create.dtype}, shape={all_targets_create.shape}")
    del all_targets_create

    # Open it in worker mode (like the worker does)
    print("\nOpening in worker mode...")
    all_targets_worker = np.lib.format.open_memmap(
        targets_path, mode='r+',
        shape=(100, 1)
    )
    print(f"Opened: dtype={all_targets_worker.dtype}, shape={all_targets_worker.shape}")

    # Write some test values (like base_target would be)
    print("\nWriting test values...")
    test_values = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65]
    for i, val in enumerate(test_values):
        print(f"  Writing {val} to index {i}")
        all_targets_worker[i] = val  # This is what the worker does

    all_targets_worker.flush()
    del all_targets_worker

    # Read back and check
    print("\nReading back...")
    all_targets_read = np.load(targets_path, mmap_mode='r')
    print(f"Read: dtype={all_targets_read.dtype}, shape={all_targets_read.shape}")
    print(f"Values: {all_targets_read[:len(test_values)].flatten()}")
    print(f"Range: [{all_targets_read[:len(test_values)].min()}, {all_targets_read[:len(test_values)].max()}]")

    # Test with numpy scalars vs arrays
    print("\n\nTesting different input types...")
    all_targets_worker2 = np.lib.format.open_memmap(targets_path, mode='r+')

    # Test 1: Python float (like 0.5)
    all_targets_worker2[10] = 0.5
    print(f"After writing Python float 0.5: {all_targets_worker2[10]}")

    # Test 2: NumPy 0-d array (like sample['target'].numpy() might return)
    all_targets_worker2[11] = np.array(0.6)
    print(f"After writing numpy scalar 0.6: {all_targets_worker2[11]}")

    # Test 3: NumPy 1-d array with shape (1,)
    all_targets_worker2[12] = np.array([0.7])
    print(f"After writing numpy array [0.7]: {all_targets_worker2[12]}")

    # Test 4: Boolean (simulating what might happen if mask is written)
    all_targets_worker2[13] = True
    print(f"After writing True: {all_targets_worker2[13]}")

    all_targets_worker2[14] = False
    print(f"After writing False: {all_targets_worker2[14]}")

    all_targets_worker2.flush()

    print("\n\nFinal check:")
    final = np.load(targets_path, mmap_mode='r')
    print(f"Values 10-14: {final[10:15].flatten()}")
    print(f"Are values 13-14 exactly 0 and 1?: {final[13,0] == 1.0 and final[14,0] == 0.0}")

    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print("\n✓ Test completed, cleaned up")

if __name__ == "__main__":
    test_memmap_write()
