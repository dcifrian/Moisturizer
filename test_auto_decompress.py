#!/usr/bin/env python3
"""Test automatic NPZ decompression"""

from Moisturizer import _get_uncompressed_npz_path
import numpy as np
from pathlib import Path

print("Testing automatic decompression...")
print()

# Test with compressed file
compressed_path = "test_data/precomputed_sequences.npz"
print(f"Input: {compressed_path}")
print()

result_path = _get_uncompressed_npz_path(compressed_path)

print()
print(f"Output: {result_path}")
print()

# Verify the file exists and can be loaded
uncompressed_path = Path(result_path)
if uncompressed_path.exists():
    print(f"✓ File exists: {uncompressed_path.name}")
    print(f"  Size: {uncompressed_path.stat().st_size / 1024:.1f} KB")

    # Verify it can be loaded with mmap_mode
    data = np.load(result_path, mmap_mode='r')
    print(f"  Arrays: {list(data.keys())}")
    print(f"  Features shape: {data['features'].shape}")
    data.close()
    print()
    print("✓ Automatic decompression working correctly!")
else:
    print("✗ File not found!")
