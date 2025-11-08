"""
Test the fixed Moisturizer.py data collection
"""
from datetime import datetime, timedelta
import sys

# Import the fixed collector
sys.path.insert(0, '/home/user/Moisturizer')
from Moisturizer import MeteoGaliciaCollector

print("=" * 60)
print("Testing Fixed Data Collection")
print("=" * 60)

# Create collector with test directory
collector = MeteoGaliciaCollector(cache_dir="./test_fixed_data")

# Get first 2 stations for quick test
print("\n1. Getting stations...")
stations_df = collector.get_all_stations()
test_stations = stations_df['station_id'].head(2).tolist()
print(f"Testing with stations: {test_stations}")

# Test fetching 7 days of data
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

print(f"\n2. Fetching historical data...")
print(f"   From: {start_date.strftime('%d/%m/%Y')}")
print(f"   To: {end_date.strftime('%d/%m/%Y')}")
print(f"   Stations: {len(test_stations)}")

parameters = [
    'HS_CV_AVG_-0.2m',  # Soil moisture
    'PP_SUM_1.5m',      # Precipitation
    'TA_AVG_1.5m',      # Temperature
    'HR_AVG_1.5m',      # Humidity
]

timeseries_df = collector.build_historical_dataset(
    station_ids=test_stations,
    parameter_ids=parameters,
    start_date=start_date,
    end_date=end_date,
    chunk_days=30
)

print("\n3. Results:")
if not timeseries_df.empty:
    print(f"✓ SUCCESS! Collected {len(timeseries_df)} records")
    print(f"\nDataFrame info:")
    print(timeseries_df.info())
    print(f"\nFirst few rows:")
    print(timeseries_df.head(10))
    print(f"\nDate range: {timeseries_df['date'].min()} to {timeseries_df['date'].max()}")
    print(f"Unique stations: {timeseries_df['station_id'].unique()}")
    print(f"Unique parameters: {timeseries_df['parameter_code'].unique()}")
else:
    print("✗ FAILED: No data collected")

print("\n" + "=" * 60)
