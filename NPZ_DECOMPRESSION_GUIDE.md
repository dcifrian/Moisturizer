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
    data_dir="meteogalicia_data/data",
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
# Test with the small dataset first
python -c "
from Moisturizer import SoilMoistureSequenceDataset
from torch.utils.data import DataLoader

ds = SoilMoistureSequenceDataset(
    data_dir='meteogalicia_data/data',
    seq_length=96,
    precomputed_path='test_data/precomputed_sequences_uncompressed.npz'
)

loader = DataLoader(ds, batch_size=16, num_workers=15)
batch = next(iter(loader))
print(f'✓ Successfully loaded batch with 15 workers!')
print(f'  Features shape: {batch[\"features\"].shape}')
"
```

## Next Steps

1. Decompress your existing datasets
2. Update file paths in your training scripts
3. Test with 15 workers to verify no errors
4. Confirm no OOM issues with the large 240GB dataset

The solution preserves all pre-normalized data and metadata without requiring regeneration.
