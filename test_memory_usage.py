#!/usr/bin/env python3
"""
Memory usage test - adapt the paths to your actual data location

Usage:
    python test_memory_usage.py /path/to/data_dir /path/to/precomputed.npz

Example:
    python test_memory_usage.py meteogalicia_data meteogalicia_data/precomputed_sequences.npz
"""

import os
import sys
import gc
import numpy as np

# Try to use psutil for memory tracking
try:
    import psutil
    process = psutil.Process()
    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024
    HAS_PSUTIL = True
    print("✓ psutil available - will track memory usage")
except ImportError:
    HAS_PSUTIL = False
    print("⚠ psutil not available - install with: pip install psutil")
    print("  Will track steps without memory measurements")
    def get_memory_mb():
        return None

def log_memory(step):
    """Log current memory usage"""
    if HAS_PSUTIL:
        mem = get_memory_mb()
        print(f"[MEMORY] {step}: {mem:.1f} MB ({mem/1024:.2f} GB)")
    else:
        print(f"[STEP] {step}")
    sys.stdout.flush()

# Patch np.load to track calls
original_load = np.load
load_calls = []

def tracked_load(file, *args, **kwargs):
    basename = os.path.basename(str(file))
    mmap_mode = kwargs.get('mmap_mode', None)

    log_memory(f"  → np.load('{basename}', mmap_mode={mmap_mode})")
    result = original_load(file, *args, **kwargs)
    log_memory(f"  ← np.load('{basename}') returned")

    load_calls.append({
        'file': basename,
        'mmap_mode': mmap_mode,
        'keys': list(result.keys()) if hasattr(result, 'keys') else None
    })

    return result

np.load = tracked_load

log_memory("START - Before imports")

from Moisturizer import SoilMoistureSequenceDataset

log_memory("After importing Moisturizer")

# Get data paths from command line or use defaults
if len(sys.argv) >= 3:
    data_dir = sys.argv[1]
    precomputed_path = sys.argv[2]
else:
    print("\nUsage: python test_memory_usage.py DATA_DIR PRECOMPUTED_PATH")
    print("\nExample:")
    print("  python test_memory_usage.py meteogalicia_data meteogalicia_data/precomputed_sequences.npz")
    print("\nUsing default test paths...")
    data_dir = "meteogalicia_data"
    precomputed_path = "test_data/precomputed_sequences.npz"

timeseries_path = os.path.join(data_dir, "raw_timeseries.csv")
stations_path = os.path.join(data_dir, "stations_metadata.csv")
nearest_path = os.path.join(data_dir, "nearest_stations.csv")

# Check files exist
missing = []
for p in [timeseries_path, stations_path, nearest_path, precomputed_path]:
    if not os.path.exists(p):
        missing.append(p)

if missing:
    print(f"\n✗ ERROR: Missing files:")
    for p in missing:
        print(f"  - {p}")
    print("\nPlease provide correct paths:")
    print("  python test_memory_usage.py DATA_DIR PRECOMPUTED_PATH")
    sys.exit(1)

print(f"\n{'='*70}")
print(f"MEMORY USAGE TEST")
print(f"{'='*70}")
print(f"Data directory: {data_dir}")
print(f"Precomputed file: {precomputed_path}")
print(f"Precomputed size: {os.path.getsize(precomputed_path) / 1024 / 1024:.1f} MB")
print(f"{'='*70}\n")

log_memory("Before creating dataset")

print("\n" + "="*70)
print("TEST 1: Load dataset (no splitting)")
print("="*70 + "\n")

dataset = SoilMoistureSequenceDataset(
    timeseries=timeseries_path,
    stations=stations_path,
    nearest=nearest_path,
    seq_length=96,
    precomputed_path=precomputed_path,
)

log_memory("After loading dataset")
print(f"\nDataset created:")
print(f"  - Length: {len(dataset)}")
print(f"  - sample_index is None: {dataset.sample_index is None}")
if hasattr(dataset, 'n_samples'):
    print(f"  - n_samples: {dataset.n_samples}")
if hasattr(dataset, 'indices'):
    print(f"  - indices: {dataset.indices}")

log_memory("Before accessing item[0]")
item = dataset[0]
log_memory("After accessing item[0]")
print(f"  - Features shape: {item['features'].shape}")

gc.collect()
log_memory("After gc.collect()")

print(f"\n{'='*70}")
print(f"TEST 2: Train/Val/Test Split")
print(f"{'='*70}\n")

log_memory("Before train_val_test_split()")

try:
    train_data, val_data, test_data = SoilMoistureSequenceDataset.train_val_test_split(
        dataset,
        val_stations_ratio=0.2,
        test_stations_ratio=0.0
    )

    log_memory("After train_val_test_split()")

    print(f"\nSplit created:")
    print(f"  - Train length: {len(train_data)}")
    print(f"  - Val length: {len(val_data) if val_data else 0}")
    print(f"  - Train sample_index is None: {train_data.sample_index is None}")
    if hasattr(train_data, 'indices'):
        print(f"  - Train indices length: {len(train_data.indices) if train_data.indices else None}")

    log_memory("Before accessing train[0]")
    train_item = train_data[0]
    log_memory("After accessing train[0]")
    print(f"  - Train features shape: {train_item['features'].shape}")

    gc.collect()
    log_memory("After gc.collect()")

except Exception as e:
    print(f"\n✗ ERROR during split: {e}")
    import traceback
    traceback.print_exc()
    log_memory("After error")

print(f"\n{'='*70}")
print(f"np.load() CALLS SUMMARY")
print(f"{'='*70}")
for i, call in enumerate(load_calls):
    print(f"{i+1}. {call['file']}")
    print(f"   mmap_mode: {call['mmap_mode']}")
    if call['keys']:
        print(f"   keys: {call['keys']}")

print(f"\n{'='*70}")
print(f"DONE")
print(f"{'='*70}")

if HAS_PSUTIL:
    final_mem = get_memory_mb()
    print(f"Final memory: {final_mem:.1f} MB ({final_mem/1024:.2f} GB)")
