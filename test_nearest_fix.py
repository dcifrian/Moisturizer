"""
Test the fixed nearest stations logic
Verify that it finds nearest stations WITH soil moisture for ALL stations
"""
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '/home/user/Moisturizer')

print("Warning: PyTorch not available. SoilMoistureSequenceDataset will not work.")
print("         Data collection functionality will still work normally.")

from Moisturizer import MeteoGaliciaCollector
import pandas as pd

print("=" * 60)
print("Testing Fixed Nearest Stations Logic")
print("=" * 60)

collector = MeteoGaliciaCollector(cache_dir="./test_nearest_fix")

# Step 1: Get stations and discover soil moisture
print("\nStep 1: Discovering stations...")
stations_df = collector.discover_stations_with_soil_moisture(force_refresh=False)

print(f"\nTotal stations: {len(stations_df)}")
print(f"Stations WITH soil moisture: {stations_df['has_soil_moisture'].sum()}")
print(f"Stations WITHOUT soil moisture: {(~stations_df['has_soil_moisture']).sum()}")

# Step 2: Calculate nearest stations
print("\n" + "=" * 60)
print("Step 2: Calculating nearest stations WITH soil moisture")
print("=" * 60)

nearest_df = collector.calculate_nearest_stations(stations_df, n_nearest=4)

# Step 3: Verify the results
print("\n" + "=" * 60)
print("Step 3: Verifying results")
print("=" * 60)

# Check a station WITH soil moisture
with_soil = stations_df[stations_df['has_soil_moisture']].iloc[0]
print(f"\nTest station WITH soil moisture: {with_soil['station_id']} - {with_soil['station_name']}")

nearest_info = nearest_df[nearest_df['station_id'] == with_soil['station_id']].iloc[0]
print(f"Nearest stations:")
for i in range(1, 5):
    nearest_id = nearest_info[f'nearest_{i}_id']
    distance = nearest_info[f'nearest_{i}_distance']
    has_sm = nearest_info[f'nearest_{i}_has_soil_moisture']
    if pd.notna(nearest_id):
        print(f"  {i}. Station {int(nearest_id)}: {distance:.0f}m, has_soil_moisture={has_sm}")
    else:
        print(f"  {i}. No station found (not enough stations with soil moisture)")

# Check a station WITHOUT soil moisture
without_soil = stations_df[~stations_df['has_soil_moisture']].iloc[0]
print(f"\nTest station WITHOUT soil moisture: {without_soil['station_id']} - {without_soil['station_name']}")

nearest_info = nearest_df[nearest_df['station_id'] == without_soil['station_id']].iloc[0]
print(f"Nearest stations WITH soil moisture:")
found_count = 0
for i in range(1, 5):
    nearest_id = nearest_info[f'nearest_{i}_id']
    distance = nearest_info[f'nearest_{i}_distance']
    has_sm = nearest_info[f'nearest_{i}_has_soil_moisture']
    if pd.notna(nearest_id):
        found_count += 1
        print(f"  {i}. Station {int(nearest_id)}: {distance:.0f}m, has_soil_moisture={has_sm}")
        # Verify it actually has soil moisture
        actual_has_sm = stations_df[stations_df['station_id'] == int(nearest_id)]['has_soil_moisture'].values[0]
        if not actual_has_sm:
            print(f"    ✗ ERROR: This station does NOT have soil moisture!")
        else:
            print(f"    ✓ Verified: Station has soil moisture")
    else:
        print(f"  {i}. No station found")

print(f"\nFound {found_count} nearest stations with soil moisture")

# Step 4: Test get_live_prediction_data
if found_count > 0:
    print("\n" + "=" * 60)
    print("Step 4: Testing get_live_prediction_data")
    print("=" * 60)

    live_data = collector.get_live_prediction_data(
        target_station_id=without_soil['station_id'],
        n_nearest=4
    )

    print(f"\nTarget station: {live_data['target_station_id']}")
    print(f"Nearby stations with soil moisture: {len(live_data['nearby_stations'])}")

    if len(live_data['nearby_stations']) > 0:
        print("\n✓ SUCCESS! Found nearby stations with soil moisture:")
        for station in live_data['nearby_stations']:
            print(f"  - Station {station['station_id']}: {station['distance']:.0f}m")
    else:
        print("\n✗ WARNING: No nearby stations with soil moisture found")

print("\n" + "=" * 60)
print("Test complete!")
