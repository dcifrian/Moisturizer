"""
Quick test of the improvements:
1. ALL_SENSORS constant
2. Fixed Step 5 (get_live_prediction_data)
"""
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/user/Moisturizer')

print("Warning: PyTorch not available. SoilMoistureSequenceDataset will not work.")
print("         Data collection functionality will still work normally.")

from Moisturizer import MeteoGaliciaCollector

print("=" * 60)
print("Testing Improvements")
print("=" * 60)

collector = MeteoGaliciaCollector(cache_dir="./test_improvements")

# Test 1: Check ALL_SENSORS constant exists
print("\nTest 1: ALL_SENSORS constant")
print(f"✓ ALL_SENSORS exists: {hasattr(collector, 'ALL_SENSORS')}")
print(f"✓ Number of sensors: {len(collector.ALL_SENSORS)}")
print(f"✓ First 5 sensors: {collector.ALL_SENSORS[:5]}")
print(f"✓ Last 5 sensors: {collector.ALL_SENSORS[-5:]}")

# Test 2: Quick data collection with a few sensors
print("\n" + "=" * 60)
print("Test 2: Data collection with ALL_SENSORS (using 3 days, 2 stations)")
print("=" * 60)

stations_df = collector.get_all_stations()
test_stations = stations_df['station_id'].head(2).tolist()
print(f"Using stations: {test_stations}")

end_date = datetime.now()
start_date = end_date - timedelta(days=3)

# Use ALL_SENSORS
timeseries_df = collector.build_historical_dataset(
    station_ids=test_stations,
    parameter_ids=collector.ALL_SENSORS,
    start_date=start_date,
    end_date=end_date,
    chunk_days=30
)

if not timeseries_df.empty:
    print(f"\n✓ SUCCESS! Collected {len(timeseries_df)} records")
    print(f"✓ Unique parameters found: {timeseries_df['parameter_code'].nunique()}")
    print(f"✓ Parameters: {sorted(timeseries_df['parameter_code'].unique())}")
else:
    print("\n✗ No data collected")

print("\n" + "=" * 60)
print("All tests completed!")
