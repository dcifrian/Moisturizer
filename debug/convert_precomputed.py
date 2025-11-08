#!/usr/bin/env python3
"""
Convert old precomputed .npz file (with pickled sample_index) to new format
This allows you to use your existing precomputed data without regenerating it!
"""

import numpy as np
import pandas as pd
from pathlib import Path

def convert_precomputed_file(old_path, new_path):
    """
    Convert old precomputed file to new pickle-free format

    Args:
        old_path: Path to old .npz file (with pickled sample_index)
        new_path: Path to save new .npz file (without pickle)
    """
    print(f"Loading old precomputed file from {old_path}...")
    print("(This will use allow_pickle=True temporarily)")

    # Load old file with pickle enabled
    old_data = np.load(old_path, allow_pickle=True)

    # Extract arrays
    features = old_data['features']
    targets = old_data['targets']
    masks = old_data['masks']
    sample_index = old_data['sample_index'].tolist()

    print(f"  Loaded {len(sample_index)} samples")
    print(f"  Features shape: {features.shape}")

    # Convert sample_index to pickle-free format
    print("Converting sample_index to pickle-free format...")
    target_stations = np.array([s['target_station'] for s in sample_index], dtype=np.int32)
    end_dates = np.array([s['end_date'].timestamp() for s in sample_index], dtype=np.float64)
    start_dates = np.array([s['start_date'].timestamp() for s in sample_index], dtype=np.float64)

    # Save new file
    print(f"Saving new precomputed file to {new_path}...")
    np.savez_compressed(
        new_path,
        features=features,
        targets=targets,
        masks=masks,
        target_stations=target_stations,
        end_dates=end_dates,
        start_dates=start_dates
    )

    print("✓ Conversion complete!")
    print(f"\nYou can now delete the old file and rename the new one:")
    print(f"  mv {new_path} {old_path}")

if __name__ == "__main__":
    from Moisturizer import MeteoGaliciaCollector

    collector = MeteoGaliciaCollector()
    old_path = collector.data_dir / "precomputed_sequences.npz"
    new_path = collector.data_dir / "precomputed_sequences_new.npz"

    if not old_path.exists():
        print(f"✗ Old precomputed file not found at {old_path}")
        print("  Nothing to convert!")
        exit(1)

    convert_precomputed_file(str(old_path), str(new_path))

    print(f"\n{'='*60}")
    print("Next steps:")
    print(f"{'='*60}")
    print(f"1. Backup your old file (optional):")
    print(f"     cp {old_path} {old_path}.backup")
    print(f"2. Replace with new file:")
    print(f"     mv {new_path} {old_path}")
    print(f"3. Run your training script as normal!")
