#!/usr/bin/env python3
"""
Time the dense array building to verify optimization works
"""

import time
from Moisturizer import buildDataset

print("=" * 70)
print("TIMING DENSE ARRAY BUILD WITH SMALL DATASET (16 days)")
print("=" * 70)

start_time = time.time()

# Run buildDataset with small dataset
train_ds, val_ds, test_ds = buildDataset(seq_length=4, days=16)

elapsed = time.time() - start_time

print("\n" + "=" * 70)
print(f"TOTAL TIME: {elapsed:.2f} seconds")
print("=" * 70)

# Estimate for full dataset
# Small dataset: 17 dates
# Full dataset: 3706 dates
# Ratio: 3706 / 17 = 218
ratio = 3706 / 17

estimated_full = elapsed * ratio
print(f"\nEstimated time for full dataset (3706 days):")
print(f"  {estimated_full:.1f} seconds = {estimated_full/60:.1f} minutes")

# The user said 30 minutes for old code
# Let's see if we improved it
old_time = 30 * 60  # 30 minutes in seconds
if estimated_full < old_time:
    speedup = old_time / estimated_full
    print(f"\nEstimated speedup: {speedup:.1f}x faster!")
else:
    print(f"\nHmm, might need more optimization...")
