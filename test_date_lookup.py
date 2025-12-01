#!/usr/bin/env python3
"""
Test if the date.normalize() fix actually works
"""
import numpy as np
import pandas as pd
from pathlib import Path

print("="*70)
print("TESTING DATE NORMALIZATION FIX")
print("="*70)

# Load dense features
dense = np.load('./meteogalicia_data/dense_features.npz')

# Convert dates like the code does
dates = pd.DatetimeIndex(dense['dates'])

print(f"\nDates from dense_features.npz:")
print(f"  Type: {type(dates)}")
print(f"  First 3 dates: {dates[:3]}")
print(f"  First date type: {type(dates[0])}")
print(f"  Timezone: {dates.tz}")
print()

# Create the lookup dict like the code does
dense_date_to_idx = {date.normalize(): idx for idx, date in enumerate(dates)}

print(f"Dense date lookup dictionary:")
print(f"  Number of keys: {len(dense_date_to_idx)}")
print(f"  First 3 keys: {list(dense_date_to_idx.keys())[:3]}")
print(f"  First key type: {type(list(dense_date_to_idx.keys())[0])}")
print(f"  First key timezone: {list(dense_date_to_idx.keys())[0].tz if hasattr(list(dense_date_to_idx.keys())[0], 'tz') else 'N/A'}")
print()

# Simulate what the dataset does
start_date = pd.Timestamp('2015-10-02')
end_date = pd.Timestamp('2015-10-03')

print(f"Simulating dataset lookup:")
print(f"  start_date: {start_date} (type: {type(start_date)}, tz: {start_date.tz})")
print(f"  end_date: {end_date} (type: {type(end_date)}, tz: {end_date.tz})")
print()

date_range = pd.date_range(start=start_date, end=end_date, freq='D')
print(f"  date_range: {date_range}")
print(f"  date_range type: {type(date_range)}")
print(f"  date_range[0] type: {type(date_range[0])}")
print(f"  date_range timezone: {date_range.tz}")
print()

# Try the lookup
print("Testing lookups:")
for date in date_range:
    normalized = date.normalize()
    idx = dense_date_to_idx.get(normalized)
    print(f"  date={date}, normalized={normalized}, found_idx={idx}")

    # Check if the normalized date is in the keys
    if normalized in dense_date_to_idx:
        print(f"    ✅ Found in dict!")
    else:
        print(f"    ❌ NOT in dict")
        # Check for any dates that are close
        for k in list(dense_date_to_idx.keys())[:5]:
            if str(k).startswith('2015-10-02') or str(k).startswith('2015-10-03'):
                print(f"    Similar key in dict: {k} (type: {type(k)})")
                print(f"    Are they equal? {k == normalized}")
                print(f"    Repr: dict_key={repr(k)}, lookup_key={repr(normalized)}")
print()

# Check if issue is timezone-related
print("Checking timezone handling:")
date_utc = pd.Timestamp('2015-10-02', tz='UTC').normalize()
date_none = pd.Timestamp('2015-10-02', tz=None).normalize()
date_auto = pd.Timestamp('2015-10-02').normalize()

print(f"  UTC normalized: {date_utc} (tz: {date_utc.tz})")
print(f"  None normalized: {date_none} (tz: {date_none.tz})")
print(f"  Auto normalized: {date_auto} (tz: {date_auto.tz})")

print(f"  In dict? UTC={date_utc in dense_date_to_idx}, None={date_none in dense_date_to_idx}, Auto={date_auto in dense_date_to_idx}")
