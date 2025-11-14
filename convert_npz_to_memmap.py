#!/usr/bin/env python3
"""
Convert NPZ file to memory-mappable format (.npy files) in chunks

NPZ files are ZIP archives and cannot be truly memory-mapped.
This extracts them to individual .npy files which CAN be memory-mapped.

Uses chunked processing to avoid OOM with large datasets.
"""

import numpy as np
from pathlib import Path
import sys
import shutil

def convert_npz_to_npy_dir(npz_path: str, output_dir: str = None, chunk_size: int = 1000):
    """
    Convert NPZ file to directory of .npy files using chunked processing

    Args:
        npz_path: Path to .npz file
        output_dir: Output directory (defaults to npz_path without .npz extension)
        chunk_size: Number of samples to process at once (for memory efficiency)
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
    print(f"  Using chunk size: {chunk_size} samples")

    # Load NPZ (just metadata, not data)
    print(f"  Loading NPZ metadata...")
    data = np.load(npz_path, mmap_mode='r')
    arrays = list(data.keys())
    print(f"  Arrays: {arrays}")

    # Process each array with chunked copy to avoid OOM
    total_size = 0
    for key in arrays:
        array = data[key]
        output_path = output_dir / f"{key}.npy"

        print(f"\n  Extracting '{key}': shape={array.shape}, dtype={array.dtype}")

        # For small arrays (< 1000 samples or 1D), copy directly
        if len(array.shape) == 1 or array.shape[0] < chunk_size:
            print(f"    Small array - copying directly...")
            np.save(output_path, np.array(array))
        else:
            # Large array - process in chunks to avoid OOM
            print(f"    Large array - processing in chunks of {chunk_size}...")

            # Create output file and write header
            n_samples = array.shape[0]

            # Use np.lib.format to write .npy file manually with chunking
            with open(output_path, 'wb') as f:
                # Write .npy header
                np.lib.format.write_array_header_2_0(f,
                    np.lib.format.header_data_from_array_1_0(np.zeros(array.shape, dtype=array.dtype)))

                # Write data in chunks
                for start_idx in range(0, n_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, n_samples)

                    if start_idx % (chunk_size * 10) == 0 or end_idx == n_samples:
                        print(f"      Progress: {end_idx}/{n_samples} ({100*end_idx/n_samples:.1f}%)")

                    # Read chunk from mmap
                    chunk = np.array(array[start_idx:end_idx])

                    # Write chunk to file
                    chunk.tofile(f)

                    # Free memory
                    del chunk

        size = output_path.stat().st_size
        total_size += size
        print(f"    ✓ Saved to {output_path.name}: {size / 1e9:.3f} GB")

    data.close()

    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print(f"\nThese .npy files can be loaded with true memory-mapping:")
    print(f"  features = np.load('{output_dir}/features.npy', mmap_mode='r')")

    return str(output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_npz_to_memmap.py <input.npz> [output_dir] [chunk_size]")
        print("\nExample:")
        print("  python convert_npz_to_memmap.py meteogalicia_data/precomputed_sequences.npz")
        print("  # Creates: meteogalicia_data/precomputed_sequences/")
        print("  #   features.npy, targets.npy, masks.npy, etc.")
        print("\nFor large datasets (to avoid OOM):")
        print("  python convert_npz_to_memmap.py data/merged_dataset.npz data/merged_dataset 500")
        print("  # Processes 500 samples at a time to limit memory usage")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    chunk_size = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    result = convert_npz_to_npy_dir(input_file, output_dir, chunk_size)
    if result:
        print(f"\n✓ Success! Use this path in your code:")
        print(f"  precomputed_path='{result}'")
    else:
        sys.exit(1)
