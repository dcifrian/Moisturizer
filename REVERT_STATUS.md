# Status: Reverted to Working Commit

This branch has been **hard reset** to commit `a448f5578ca0190b15b7349d48b10ec28f85a7dc` which is known to work correctly.

## What Works in This Commit

- ✅ Memory-mapped NPZ loading with `mmap_mode='r'`
- ✅ Works with up to 8 workers reliably
- ✅ Low RAM usage (only accessed data loaded)
- ✅ Train/val/test splits work correctly
- ✅ No zlib or BadZipFile errors with normal worker counts

## Known Limitations

- With 15+ workers, BadZipFile errors can occur (inherent limitation of compressed NPZ + many workers)
- For very high worker counts, use uncompressed `.npy` files instead

## What Was Broken in Later Commits

All commits after a448f557 attempted to "fix" issues but made things worse:
- Adding worker_init_fn caused 128GB+ RAM usage
- Lazy sample_index caused IndexError on splits
- Various other changes broke basic functionality

## Next Steps

If you need to support 15+ workers with large compressed NPZ files, we need a completely different approach - likely switching to uncompressed files or HDF5 format. But the current code works correctly for normal use cases (1-8 workers).

## Testing

Verified working with test dataset:
- 1 worker: ✅ Works
- 8 workers: ✅ Works
- Memory usage: Normal (not excessive)
