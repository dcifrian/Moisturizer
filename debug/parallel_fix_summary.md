# Parallel Processing Fix Summary

## Problem

The previous parallel implementation had a critical flaw that caused:
- **80GB RAM usage** (instead of expected ~3-4GB)
- **100% CPU usage with no progress**
- Workers constantly loading data instead of processing it

### Root Cause

Each worker process was loading the **entire dataset for every batch**:

```python
def _process_batch_worker(batch_info):
    # THIS WAS CALLED FOR EVERY BATCH!
    dataset = SoilMoistureSequenceDataset(...)  # Load full dataset
    # Process batch...
```

With 286 batches and 32 workers, the dataset was being loaded thousands of times:
- 32 workers × ~2.5GB per load = **80GB RAM** ✓
- Workers spent 99% of time loading data, 1% processing
- Each load takes ~30-60 seconds, so progress was glacially slow

The log showed this clearly:
```
Loading data files...
Loading dense feature arrays...
Building sample index...
Loading data files...           # Repeated thousands of times!
Loading dense feature arrays...
Building sample index...
...
```

## Solution

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

## Additional Improvements

1. **Conservative worker count**: Default to `min(8, cpu_count)` instead of all cores
   - 8 workers × 2.5GB ≈ **20GB RAM** (much more reasonable)
   - Still provides good parallelization
   - User can override with `num_workers` parameter if they have more RAM

2. **Memory usage reporting**: Shows estimated memory usage before starting

3. **Better documentation**: Explains the parallel strategy in docstring

## Expected Results

**Before (broken):**
- Dataset loads: ~9,000+ times (286 batches × 32 workers)
- RAM usage: 80GB
- Time: Never completes
- CPU: 100% (busy loading data)

**After (fixed):**
- Dataset loads: 8 times (once per worker)
- RAM usage: ~20GB
- Time: ~30-60 minutes (estimated)
- CPU: 100% (actually processing data)

## How to Run

```bash
python precompute_augmented.py
```

Or with custom settings:
```python
from precompute_augmented import generate_all_augmentations_batched

# Use 4 workers if RAM is limited
generate_all_augmentations_batched(num_workers=4)

# Use 16 workers if you have 40GB+ RAM available
generate_all_augmentations_batched(num_workers=16)
```

## Monitoring Progress

The script will show:
1. Initial dataset analysis
2. Worker initialization (8 workers loading dataset)
3. Progress updates: `Progress: 50/286 batches complete (17.5%)`
4. Batch merging and normalization
5. Final statistics

You should see steady progress through the batches, not stuck reloading data.
