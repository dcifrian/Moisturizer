#!/usr/bin/env python3
"""
Convert NPZ file to memory-mappable format (.npy files) using ZIP streaming

NPZ files are ZIP archives. This extracts .npy files directly from the ZIP
without loading them into RAM.
"""

import numpy as np
from pathlib import Path
import sys
import zipfile
import shutil

def convert_npz_to_npy_dir_streaming(npz_path: str, output_dir: str = None):
    """
    Convert NPZ to .npy directory by extracting from ZIP archive

    This streams files from the ZIP without loading into RAM.
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
    print(f"\nExtracting .npy files from ZIP archive (streaming mode)...")

    # Open as ZIP file
    total_size = 0
    with zipfile.ZipFile(npz_path, 'r') as zf:
        members = zf.namelist()
        print(f"  Found {len(members)} files in archive")

        for i, member in enumerate(members, 1):
            if not member.endswith('.npy'):
                continue

            print(f"\n  [{i}/{len(members)}] Extracting '{member}'...")

            # Get file info
            info = zf.getinfo(member)
            compressed_size = info.compress_size / 1e9
            uncompressed_size = info.file_size / 1e9

            print(f"    Compressed: {compressed_size:.3f} GB")
            print(f"    Uncompressed: {uncompressed_size:.3f} GB")

            # Extract directly to output directory
            # This streams from ZIP without loading into RAM
            output_path = output_dir / member

            print(f"    Extracting to {output_path}...")
            zf.extract(member, output_dir)

            actual_size = output_path.stat().st_size
            total_size += actual_size
            print(f"    ✓ Saved: {actual_size / 1e9:.3f} GB")

    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Total size: {total_size / 1e9:.2f} GB")
    print(f"\nThese .npy files can be loaded with true memory-mapping:")
    print(f"  features = np.load('{output_dir}/features.npy', mmap_mode='r')")

    return str(output_dir)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_npz_streaming.py <input.npz> [output_dir]")
        print("\nExample:")
        print("  python convert_npz_streaming.py data/merged_dataset.npz")
        print("  # Creates: data/merged_dataset/")
        print("\nThis version uses ZIP streaming to avoid loading into RAM.")
        print("Suitable for datasets larger than available RAM (e.g., 240GB dataset with 128GB RAM).")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    result = convert_npz_to_npy_dir_streaming(input_file, output_dir)
    if result:
        print(f"\n✓ Success! Use this path in your code:")
        print(f"  precomputed_path='{result}'")
    else:
        sys.exit(1)
