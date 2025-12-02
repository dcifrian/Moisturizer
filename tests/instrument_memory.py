#!/usr/bin/env python3
"""
Instrumented version to track memory usage during dataset loading
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
except ImportError:
    HAS_PSUTIL = False
    def get_memory_mb():
        return "N/A (install psutil)"

def log_memory(step):
    """Log current memory usage"""
    if HAS_PSUTIL:
        mem = get_memory_mb()
        print(f"[MEMORY] {step}: {mem:.1f} MB")
    else:
        print(f"[STEP] {step}")
    sys.stdout.flush()

# Instrument key functions
original_load = np.load

def instrumented_load(file, *args, **kwargs):
    log_memory(f"Before np.load({os.path.basename(str(file))})")
    mmap_mode = kwargs.get('mmap_mode', None)
    if mmap_mode:
        print(f"  Loading with mmap_mode='{mmap_mode}'")
    result = original_load(file, *args, **kwargs)
    log_memory(f"After np.load({os.path.basename(str(file))})")
    return result

np.load = instrumented_load

log_memory("START - Before imports")

from Moisturizer import SoilMoistureSequenceDataset

log_memory("After importing Moisturizer")

# Check if we have a precomputed file to test with
if not os.path.exists('test_data/precomputed_sequences.npz'):
    print("ERROR: test_data/precomputed_sequences.npz not found")
    sys.exit(1)

log_memory("Before loading dataset")

# Test loading WITHOUT creating splits first
print("\n" + "="*70)
print("TEST 1: Load dataset without splitting")
print("="*70 + "\n")

dataset = SoilMoistureSequenceDataset(
    timeseries="meteogalicia_data/raw_timeseries.csv",
    stations="meteogalicia_data/stations_metadata.csv",
    nearest="meteogalicia_data/nearest_stations.csv",
    seq_length=96,
    precomputed_path="test_data/precomputed_sequences.npz",
)

log_memory("After loading dataset")
print(f"Dataset length: {len(dataset)}")

log_memory("Before accessing first item")
item = dataset[0]
log_memory("After accessing first item")
print(f"Features shape: {item['features'].shape}")

# Force garbage collection
gc.collect()
log_memory("After gc.collect()")

print("\n" + "="*70)
print("TEST 2: Test train_val_test_split")
print("="*70 + "\n")

log_memory("Before train_val_test_split()")

try:
    train_data, val_data, test_data = dataset.train_val_test_split(
        dataset,
        val_stations_ratio=0.2,
        test_stations_ratio=0.0
    )

    log_memory("After train_val_test_split()")
    print(f"Train length: {len(train_data)}")
    print(f"Val length: {len(val_data) if val_data else 0}")

    log_memory("Before accessing train item")
    train_item = train_data[0]
    log_memory("After accessing train item")
    print(f"Train features shape: {train_item['features'].shape}")

    gc.collect()
    log_memory("After gc.collect()")

except Exception as e:
    print(f"ERROR during split: {e}")
    import traceback
    traceback.print_exc()
    log_memory("After error")

print("\n" + "="*70)
print("DONE")
print("="*70)
