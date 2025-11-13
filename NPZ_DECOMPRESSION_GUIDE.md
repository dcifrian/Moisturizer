# NPZ Decompression Guide

## Problem

Compressed NPZ files (created with `np.savez_compressed()`) cause zlib errors when accessed by multiple DataLoader workers simultaneously:

```
zlib.error: Error -3 while decompressing data: invalid stored block lengths
zipfile.BadZipFile: Overlapped entries: 'masks.npy' (possible zip bomb)
```

This is a fundamental limitation of Python's zipfile module when many processes try to read from the same compressed archive.

## Solution

Convert compressed NPZ files to uncompressed format for true memory-mapping without decompression overhead.

### Step 1: Decompress Your Dataset

```bash
# Decompress the precomputed dataset
python decompress_npz.py meteogalicia_data/precomputed_sequences.npz

# This creates: meteogalicia_data/precomputed_sequences_uncompressed.npz
```

For the large augmented dataset:
```bash
python decompress_npz.py data/batches/merged_dataset.npz
# Creates: data/batches/merged_dataset_uncompressed.npz
```

### Step 2: Update Your Code

Use the uncompressed file path:

```python
dataset = SoilMoistureSequenceDataset(
    timeseries="meteogalicia_data/raw_timeseries.csv",
    stations="meteogalicia_data/stations_metadata.csv",
    nearest="meteogalicia_data/nearest_stations.csv",
    seq_length=96,
    precomputed_path="meteogalicia_data/precomputed_sequences_uncompressed.npz",  # Changed
    # ... other parameters
)
```

### Step 3: Test with Multiple Workers

```python
train_loader = DataLoader(
    train_data,
    batch_size=512,
    num_workers=15,  # Now works without zlib errors!
    persistent_workers=True,
    shuffle=True
)
```

## Technical Details

### Why This Works

- **Compressed NPZ**: ZIP archive requiring decompression on every access
  - `np.load(path, mmap_mode='r')` still decompresses data
  - Multiple workers cause concurrent decompression → zlib errors

- **Uncompressed NPZ**: Direct memory-mapped file access
  - `np.load(path, mmap_mode='r')` truly memory-maps without decompression
  - Unlimited workers can read simultaneously
  - No RAM overhead (only maps pages as needed)

### Disk Space Trade-off

- Small dataset: 542 MB → ~11 GB (21x increase)
- Large augmented dataset: ~11 GB → ~240 GB (21x increase)

The user confirmed disk space is not a concern, avoiding the need to regenerate datasets (which takes hours).

### Compression Ratio

The `decompress_npz.py` utility reports the compression ratio:

```
  Input size: 0.54 GB
  Uncompressed size: 11.38 GB
  Output size: 11.38 GB
  Compression ratio was: 21.30x
```

## How precompute_augmented.py Avoided This

The `precompute_augmented.py` script successfully used 32 processes because:

1. **Generation phase**: Uses `np.memmap()` (uncompressed disk-backed arrays)
2. **Storage phase**: Saves to compressed NPZ only for final archival
3. **Runtime usage**: Never loads compressed NPZ with `mmap_mode='r'`

This is the pattern we're now following: use uncompressed files at runtime, compress only for archival if needed.

## Verification

After decompression, verify it works:

```bash
# Quick verification test - checks mmap access from multiple processes
python3 << 'EOF'
import numpy as np
from multiprocessing import Pool

def test_access(worker_id):
    """Each worker loads and accesses the uncompressed NPZ"""
    data = np.load('test_data/precomputed_sequences_uncompressed.npz', mmap_mode='r')
    # Access some data to verify no zlib errors
    features = data['features'][0]
    return f"Worker {worker_id}: OK (shape={features.shape})"

print("Testing multi-process access to uncompressed NPZ...")
with Pool(15) as pool:
    results = pool.map(test_access, range(15))
    for r in results:
        print(r)
print("\n✓ All 15 workers accessed the file successfully!")
print("✓ No zlib errors with uncompressed NPZ!")
EOF
```

## Next Steps

1. Decompress your existing datasets
2. Update file paths in your training scripts
3. Test with 15 workers to verify no errors
4. Confirm no OOM issues with the large 240GB dataset

The solution preserves all pre-normalized data and metadata without requiring regeneration.
