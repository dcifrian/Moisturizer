#!/usr/bin/env python3
"""
Detailed instrumentation of __getitem__ to find the memory leak
"""

import os
import sys
import gc
import numpy as np
import torch

try:
    import psutil
    process = psutil.Process()
    def get_memory_mb():
        return process.memory_info().rss / 1024 / 1024
except ImportError:
    print("ERROR: Install psutil: pip install psutil")
    sys.exit(1)

def log_memory(step):
    mem = get_memory_mb()
    print(f"[MEMORY] {step}: {mem:.1f} MB ({mem/1024:.2f} GB)")
    sys.stdout.flush()

log_memory("START")

from Moisturizer import SoilMoistureSequenceDataset

log_memory("After import")

# Load dataset
print("\nLoading dataset...")
dataset = SoilMoistureSequenceDataset(
    timeseries="meteogalicia_data/raw_timeseries.csv",
    stations="meteogalicia_data/stations_metadata.csv",
    nearest="meteogalicia_data/nearest_stations.csv",
    seq_length=96,
    precomputed_path="meteogalicia_data/precomputed_sequences.npz",
)

log_memory("After dataset load")

print(f"\nDataset loaded: {len(dataset)} samples")
print(f"Type of precomputed_data: {type(dataset.precomputed_data)}")
print(f"Type of features: {type(dataset.precomputed_data['features'])}")

# Now manually walk through __getitem__ step by step
print("\n" + "="*70)
print("STEP-BY-STEP __getitem__ INSTRUMENTATION")
print("="*70)

idx = 0
actual_idx = dataset._indices[idx] if hasattr(dataset, '_indices') and dataset._indices is not None else idx

log_memory("Before accessing features[idx]")

# Access features
features = dataset.precomputed_data['features'][actual_idx]
log_memory(f"After accessing features[{actual_idx}]")
print(f"  features type: {type(features)}, shape: {features.shape}, dtype: {features.dtype}")

# Access target
target = dataset.precomputed_data['targets'][actual_idx]
log_memory(f"After accessing targets[{actual_idx}]")
print(f"  target type: {type(target)}, shape: {target.shape}, dtype: {target.dtype}")

# Access mask
mask = dataset.precomputed_data['masks'][actual_idx]
log_memory(f"After accessing masks[{actual_idx}]")
print(f"  mask type: {type(mask)}, shape: {mask.shape}, dtype: {mask.dtype}")

# Convert to tensors
log_memory("Before torch.from_numpy(features)")
features_tensor = torch.from_numpy(features)
log_memory("After torch.from_numpy(features)")
print(f"  features_tensor type: {type(features_tensor)}, shape: {features_tensor.shape}")

log_memory("Before torch.from_numpy(target)")
target_tensor = torch.from_numpy(target)
log_memory("After torch.from_numpy(target)")

log_memory("Before torch.from_numpy(mask)")
mask_tensor = torch.from_numpy(mask)
log_memory("After torch.from_numpy(mask)")

# Check memory-mapped array info
print(f"\n" + "="*70)
print("MEMORY-MAPPED ARRAY DETAILS")
print("="*70)
print(f"features array flags:")
print(f"  C_CONTIGUOUS: {dataset.precomputed_data['features'].flags['C_CONTIGUOUS']}")
print(f"  OWNDATA: {dataset.precomputed_data['features'].flags['OWNDATA']}")
print(f"  WRITEABLE: {dataset.precomputed_data['features'].flags['WRITEABLE']}")

# Check if it's actually a memmap
print(f"\nIs memmap? {isinstance(dataset.precomputed_data['features'], np.memmap)}")
print(f"Base type: {type(dataset.precomputed_data['features'].base)}")

# Check slice
print(f"\nSlice [0] flags:")
print(f"  OWNDATA: {features.flags['OWNDATA']}")
print(f"Is slice a memmap? {isinstance(features, np.memmap)}")

gc.collect()
log_memory("After gc.collect()")

print("\n" + "="*70)
print("TESTING SECOND ACCESS")
print("="*70)

idx2 = 1
actual_idx2 = dataset._indices[idx2] if hasattr(dataset, '_indices') and dataset._indices is not None else idx2

log_memory(f"Before accessing item {idx2}")
features2 = dataset.precomputed_data['features'][actual_idx2]
log_memory(f"After accessing features[{actual_idx2}]")

gc.collect()
log_memory("After gc.collect()")

print("\n" + "="*70)
print("DONE")
print("="*70)
