#!/usr/bin/env python3
"""
Convert NPZ file to memory-mappable format (.npy files or .dat files)

NPZ files are ZIP archives and cannot be truly memory-mapped.
This extracts them to individual .npy files which CAN be memory-mapped.
"""

import numpy as np
from pathlib import Path
import sys
import shutil

def convert_npz_to_npy_dir(npz_path: str, output_dir: str = None):
    """
    Convert NPZ file to directory of .npy files

    Args:
        npz_path: Path to .npz file
        output_dir: Output directory (defaults to npz_path without .npz extension)
    """
    npz_path = Path(npz_path)

    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found")
        return False

    if output_dir is None:
        output_dir = npz_path.parent / npz_path.stem
    else:
        output_dir = Path(output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {npz_path}")
    print(f"  to directory: {output_dir}")
    print(f"  Input size: {npz_path.stat().st_size / 1e9:.2f} GB")

    # Load NPZ
    print(f"  Loading NPZ file...")
    data = np.load(npz_path)
    arrays = list(data.keys())
    print(f"  Arrays: {arrays}")

    # Extract each array to individual .npy file
    total_size = 0
    for key in arrays:
        array = data[key]
        output_path = output_dir / f"{key}.npy"

        print(f"  Extracting '{key}': shape={array.shape}, dtype={array.dtype}")
        np.save(output_path, array)

        size = output_path.stat().st_size
        total_size += size
        print(f"    Saved to {output_path.name}: {size / 1e9:.3f} GB")

    data.close()

    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print(f"\nThese .npy files can be loaded with true memory-mapping:")
    print(f"  features = np.load('{output_dir}/features.npy', mmap_mode='r')")

    return str(output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_npz_to_memmap.py <input.npz> [output_dir]")
        print("\nExample:")
        print("  python convert_npz_to_memmap.py meteogalicia_data/precomputed_sequences.npz")
        print("  # Creates: meteogalicia_data/precomputed_sequences/")
        print("  #   features.npy, targets.npy, masks.npy, etc.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    result = convert_npz_to_npy_dir(input_file, output_dir)
    if result:
        print(f"\n✓ Success! Use this path in your code:")
        print(f"  precomputed_path='{result}'")
    else:
        sys.exit(1)
