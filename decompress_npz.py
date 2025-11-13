#!/usr/bin/env python3
"""
Decompress a compressed NPZ file to uncompressed format for true memory-mapping.

This allows using the dataset with many DataLoader workers without zlib errors.
"""
import numpy as np
from pathlib import Path
import sys


def decompress_npz(input_path: str, output_path: str = None):
    """
    Decompress a compressed NPZ file to uncompressed format.

    Args:
        input_path: Path to compressed .npz file
        output_path: Path for uncompressed output (defaults to input_path with '_uncompressed' suffix)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: {input_path} not found!")
        return False

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_uncompressed.npz"
    else:
        output_path = Path(output_path)

    print(f"Decompressing {input_path}...")
    print(f"  Input size: {input_path.stat().st_size / 1e9:.2f} GB")

    # Load compressed file
    print(f"  Loading compressed data...")
    data = np.load(input_path)

    # Get all array names
    arrays = list(data.keys())
    print(f"  Arrays: {arrays}")

    # Calculate uncompressed size
    total_size = sum(data[key].nbytes for key in arrays)
    print(f"  Uncompressed size: {total_size / 1e9:.2f} GB")

    # Save uncompressed (no _compressed suffix means uncompressed)
    print(f"  Saving uncompressed to {output_path}...")
    print(f"  This may take a few minutes...")

    # Create dict of arrays
    arrays_dict = {key: data[key] for key in arrays}

    # Save WITHOUT compression
    np.savez(output_path, **arrays_dict)

    data.close()

    output_size = output_path.stat().st_size / 1e9
    print(f"  Output size: {output_size:.2f} GB")
    print(f"  Compression ratio was: {output_size / (input_path.stat().st_size / 1e9):.2f}x")
    print(f"\n✓ Decompression complete!")
    print(f"\nUsage:")
    print(f"  dataset = SoilMoistureSequenceDataset(")
    print(f"      ...,")
    print(f"      precomputed_path='{output_path}',  # Use uncompressed file")
    print(f"      ...")
    print(f"  )")

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python decompress_npz.py <input.npz> [output.npz]")
        print("\nExample:")
        print("  python decompress_npz.py meteogalicia_data/precomputed_sequences.npz")
        print("  # Creates: meteogalicia_data/precomputed_sequences_uncompressed.npz")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = decompress_npz(input_file, output_file)
    sys.exit(0 if success else 1)
