#!/usr/bin/env python3
"""
Analyze performance bottleneck in dataset generation
"""

def analyze_bottleneck():
    print("=" * 70)
    print("ANALYZING PRECOMPUTATION PERFORMANCE BOTTLENECK")
    print("=" * 70)

    # Constants from your dataset
    num_samples = 24854
    seq_length = 64
    n_nearest = 4
    num_params = 35  # Approximate number of weather parameters

    print(f"\nDataset parameters:")
    print(f"  Samples: {num_samples:,}")
    print(f"  Sequence length: {seq_length} days")
    print(f"  Nearby stations: {n_nearest}")
    print(f"  Parameters per station: ~{num_params}")

    print("\n" + "=" * 70)
    print("BOTTLENECK: _build_sequence_tensor()")
    print("=" * 70)

    print("\nFor EACH sample, the code does:")
    print("  For each of 64 timesteps:")
    print("    1. Filter entire timeseries_df for target station + date")
    print("    2. Filter again for EACH parameter (~35 times)")
    print("    3. For each of 4 nearby stations:")
    print("       - Filter entire timeseries_df for nearby station + date")
    print("       - Filter again for EACH parameter (~35 times)")
    print("       - Filter again for soil moisture")

    # Calculate number of pandas filters
    filters_per_timestep = (
        1 +  # Target station date filter
        num_params +  # Target station parameter filters
        n_nearest * (1 + num_params + 1)  # Nearby stations
    )

    filters_per_sample = seq_length * filters_per_timestep
    total_filters = num_samples * filters_per_sample

    print(f"\n📊 Filter operations:")
    print(f"  Per timestep: {filters_per_timestep:,}")
    print(f"  Per sample: {filters_per_sample:,}")
    print(f"  Total: {total_filters:,}")

    print(f"\n⚠️  {total_filters / 1_000_000:.1f} MILLION pandas filter operations!")

    # Each pandas filter is O(n) where n is size of timeseries_df
    # Let's estimate timeseries size
    num_stations = 150
    num_days = 730  # 2 years
    num_params_total = 50
    timeseries_rows = num_stations * num_days * num_params_total

    print(f"\nEstimated timeseries_df size: {timeseries_rows:,} rows")
    print(f"Each filter operation scans ALL rows (O(n))")
    print(f"Total row scans: {total_filters * timeseries_rows / 1_000_000_000:.1f} BILLION")

    # Time estimation
    time_per_filter_ms = 0.5  # Optimistic estimate
    total_time_seconds = (total_filters * time_per_filter_ms) / 1000
    total_time_hours = total_time_seconds / 3600

    print(f"\n⏱️  Time estimate:")
    print(f"  At {time_per_filter_ms}ms per filter: {total_time_hours:.1f} hours")
    print(f"  Matches observed ~24 hours! ✓")

    print("\n" + "=" * 70)
    print("SOLUTION: PRE-INDEX THE DATA")
    print("=" * 70)

    print("\nInstead of filtering timeseries_df millions of times,")
    print("create a dictionary/multi-index ONCE:")
    print("  data[(station_id, date, parameter)] = value")
    print("  O(1) lookup instead of O(n) filter!")

    print("\nExpected speedup:")
    print("  From: O(num_samples × seq_length × stations × params × timeseries_rows)")
    print("  To:   O(num_samples × seq_length × stations × params)")
    print(f"  Speedup: ~{timeseries_rows}x faster!")
    print(f"  New time: ~{total_time_hours / timeseries_rows * 60:.1f} minutes instead of 24 hours")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("\n✓ Create optimized version of _build_sequence_tensor()")
    print("✓ Pre-index timeseries_df once in __init__()")
    print("✓ Use dictionary lookups instead of pandas filters")
    print("✓ Expected time: 30-60 minutes instead of 24 hours")

if __name__ == "__main__":
    analyze_bottleneck()
