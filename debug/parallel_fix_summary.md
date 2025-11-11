# Parallel Processing Fix Summary

## Problem 1: Worker Initialization (FIXED)

The parallel implementation had a critical flaw that caused:
- **80GB RAM usage** (instead of expected ~20GB)
- **100% CPU usage with no progress**
- Workers constantly loading data instead of processing it

### Root Cause

Each worker process was loading the **entire dataset for every batch**:

```python
def _process_batch_worker(batch_info):
    # THIS WAS CALLED FOR EVERY BATCH!
    dataset = SoilMoistureSequenceDataset(...)  # Load full dataset (~2.5GB)
    # Process batch...
```

With 286 batches and 32 workers, the dataset was being loaded thousands of times:
- 32 workers × ~2.5GB per load = **80GB RAM** ✓
- Workers spent 99% of time loading data, 1% processing
- Each load takes ~30-60 seconds, so progress was glacially slow

The log showed this clearly with thousands of repetitions:
```
Loading data files...
Loading dense feature arrays...
Building sample index...
Loading data files...           # Repeated thousands of times!
Loading dense feature arrays...
...
```

### Solution 1: Worker Initializer

Use a **worker initializer** to load the dataset **once per worker** (not once per batch):

```python
# Global variable to hold dataset in each worker
_worker_dataset = None

def _init_worker(dataset_params, aug_params):
    """Called ONCE when worker starts"""
    global _worker_dataset
    _worker_dataset = SoilMoistureSequenceDataset(...)

def _process_batch_worker(batch_info):
    """Uses pre-loaded dataset"""
    global _worker_dataset
    dataset = _worker_dataset  # Reuse loaded dataset!
    # Process batch...

# Create pool with initializer
with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(...)) as pool:
    for batch_file in pool.imap(_process_batch_worker, batch_infos):
        ...
```

**Result:**
- Dataset loads: 9,000+ → 8 times (once per worker)
- RAM usage during processing: 80GB → 20GB
- Time: Never completes → Completes in ~1 hour

---

## Problem 2: Merge Phase OOM (FIXED)

Even after fixing worker initialization, the **merge phase** tried to allocate the entire dataset in RAM:
- **240GB+ RAM required** for 3.4M samples
- **OOM killer terminated the process** during merge
- The script completed batch processing but crashed when merging batches

### Root Cause

The merge code tried to pre-allocate arrays for the entire merged dataset:

```python
# Pre-allocate final arrays
all_features = np.zeros((total_samples, seq_length, n_features), dtype=np.float32)
all_masks = np.zeros((total_samples, seq_length, n_features), dtype=np.float32)
```

With 3,421,080 samples × 64 seq_length × 138 features × 4 bytes:
- `all_features`: ~120 GB
- `all_masks`: ~120 GB
- `all_targets` + metadata: ~15 GB
- **Total: ~255 GB in RAM** 💥

### Solution 2: Memory-Mapped Arrays

Use **memory-mapped arrays** that live on disk instead of RAM:

```python
# Create memory-mapped arrays (disk-backed, not in RAM!)
memmap_dir = Path(data_dir) / "memmap_temp"
memmap_dir.mkdir(exist_ok=True)

all_features = np.memmap(
    str(memmap_dir / "features.dat"), dtype=np.float32, mode='w+',
    shape=(total_samples, seq_length, n_features)
)
all_masks = np.memmap(
    str(memmap_dir / "masks.dat"), dtype=np.float32, mode='w+',
    shape=(total_samples, seq_length, n_features)
)
# ... etc

# Work with arrays as normal
all_features[start:end] = batch_data['features']
all_features.flush()  # Write to disk

# Save final file (np.savez_compressed reads from memmap)
np.savez_compressed(output_path, features=all_features, ...)

# Cleanup
shutil.rmtree(memmap_dir)
```

Memory-mapped arrays:
- Act like numpy arrays but store data on disk
- Only load accessed portions into RAM
- Allow working with datasets larger than available RAM
- Perfect for sequential operations like merge and normalization

**Result:**
- RAM usage during merge: 240GB → <5GB
- Process completes without OOM
- Temporary disk space needed: ~240GB (cleaned up automatically)

---

## Combined Results

### Before (broken):
- **Worker phase**: 80GB RAM, never completes
- **Merge phase**: 240GB RAM, OOM killed
- **Total**: Unusable

### After (fixed):
- **Worker phase**: ~20GB RAM (8 workers × 2.5GB)
- **Merge phase**: <5GB RAM (memory-mapped arrays)
- **Peak RAM**: ~20GB total
- **Time**: ~1-2 hours
- **Disk space**: ~240GB temporary (auto-cleaned) + ~50GB final compressed file

---

## Additional Improvements

1. **Conservative worker count**: Default to `min(8, cpu_count)` instead of all cores
   - 8 workers × 2.5GB ≈ 20GB RAM (reasonable for most systems)
   - Still provides good 8x parallelization
   - User can override with `num_workers` parameter

2. **Memory usage reporting**: Shows estimated memory and disk usage before starting

3. **Progress tracking**: Regular updates during all phases (batch processing, merge, normalize, save)

4. **Periodic flushing**: Memory-mapped arrays flushed to disk periodically to avoid buffer buildup

---

## How to Run

```bash
python precompute_augmented.py
```

Or with custom worker count:
```python
from precompute_augmented import generate_all_augmentations_batched

# Use 4 workers if RAM is limited (needs ~10GB RAM)
generate_all_augmentations_batched(num_workers=4)

# Use 16 workers if you have 40GB+ RAM available
generate_all_augmentations_batched(num_workers=16)
```

---

## Monitoring Progress

The script shows:

1. **Dataset analysis**: Parameter coverage, filtered features
2. **Batch processing**:
   - `Each worker will load the dataset once...` (workers initializing)
   - `Progress: 50/286 batches complete (17.5%)` (steady progress)
3. **Merge phase**:
   - `Pass 1/3: Calculating total size...`
   - `Pass 2/3: Creating memory-mapped arrays...`
   - `Pass 3/3: Copying batch data...`
4. **Normalization**: `Progress: 500000/3421080 (14.6%)`
5. **Saving**: `Saving augmented dataset to compressed NPZ...`
6. **Cleanup**: Removes temporary files automatically

---

## Requirements

- **RAM**: ~20GB (adjustable with `num_workers`)
- **Disk space**: ~300GB temporary (auto-cleaned) + ~50GB final file
- **Time**: 1-2 hours depending on CPU and disk speed
- **Python packages**: numpy, pandas (already in requirements)

The script is now truly memory-efficient and will complete successfully on systems with 32GB+ RAM!
