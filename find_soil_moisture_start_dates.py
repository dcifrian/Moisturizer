#!/usr/bin/env python3
"""
Find the earliest date each station started having soil moisture data.

Uses binary search to efficiently find the boundary between no-data and data.
"""

import pandas as pd
from datetime import datetime, timedelta
from Moisturizer import MeteoGaliciaCollector
import time


def has_data_on_date(collector, station_id, date, parameter='HS_CV_AVG_-0.2m'):
    """
    Check if a station has data for a parameter on a specific date.

    Args:
        collector: MeteoGaliciaCollector instance
        station_id: Station ID to check
        date: datetime object for the date to check
        parameter: Parameter code (default: soil moisture)

    Returns:
        bool: True if data exists for that date, False otherwise
    """
    # Format date as dd/MM/yyyy
    date_str = date.strftime('%d/%m/%Y')

    # Request just this one day for this one station and parameter
    response = collector.get_daily_data(
        station_ids=[station_id],
        parameter_ids=[parameter],
        start_date=date_str,
        end_date=date_str
    )

    # Parse response
    if response is None:
        return False

    df = collector.parse_data_to_dataframe(response)

    # Check if we got any valid data
    if df.empty:
        return False

    # Check if there's actual non-null data for this parameter
    data = df[
        (df['station_id'] == station_id) &
        (df['parameter_code'] == parameter) &
        (df['value'].notna())
    ]

    return len(data) > 0


def binary_search_start_date(collector, station_id, earliest_possible, latest_known_good, parameter='HS_CV_AVG_-0.2m'):
    """
    Use binary search to find the earliest date a station has data.

    Args:
        collector: MeteoGaliciaCollector instance
        station_id: Station ID to search
        earliest_possible: datetime - earliest date to search (lower bound)
        latest_known_good: datetime - latest date known to have data (upper bound)
        parameter: Parameter code

    Returns:
        datetime: The earliest date with data, or None if no data found
    """
    print(f"\n  Binary search for station {station_id}:")
    print(f"    Range: {earliest_possible.date()} to {latest_known_good.date()}")

    # Check if latest_known_good actually has data
    if not has_data_on_date(collector, station_id, latest_known_good, parameter):
        print(f"    No data even on {latest_known_good.date()}")
        return None

    # Check if earliest_possible has data (already at boundary)
    time.sleep(0.1)  # Rate limiting
    if has_data_on_date(collector, station_id, earliest_possible, parameter):
        print(f"    Data exists from earliest date {earliest_possible.date()}")
        return earliest_possible

    # Binary search
    left = earliest_possible
    right = latest_known_good
    result = None
    iterations = 0

    while left <= right:
        iterations += 1
        mid = left + (right - left) // 2

        print(f"    Iteration {iterations}: checking {mid.date()}...", end=' ')
        time.sleep(0.5)  # Rate limiting to be nice to the API

        if has_data_on_date(collector, station_id, mid, parameter):
            # Data exists, try earlier
            result = mid
            right = mid - timedelta(days=1)
            print("✓ has data, searching earlier")
        else:
            # No data, try later
            left = mid + timedelta(days=1)
            print("✗ no data, searching later")

    return result


def find_all_start_dates(test_single_station=None):
    """
    Find start dates for all stations with soil moisture.

    Args:
        test_single_station: If provided, only test this station ID
    """
    print("=" * 60)
    print("FINDING SOIL MOISTURE DATA START DATES")
    print("=" * 60)

    # Initialize collector
    collector = MeteoGaliciaCollector()

    # Load stations
    stations_df = pd.read_csv(collector.stations_file)

    # Filter to stations with soil moisture
    soil_stations = stations_df[stations_df['has_soil_moisture'] == True]

    if test_single_station is not None:
        soil_stations = soil_stations[soil_stations['station_id'] == test_single_station]
        print(f"\nTesting single station: {test_single_station}")

    print(f"\nFound {len(soil_stations)} stations with soil moisture sensors")

    # Define search range
    # Try going back further to see if there's data before 2010
    earliest_possible = datetime(2005, 1, 1)
    # Use today as the latest known good date
    latest_known_good = datetime.now()

    results = []

    for idx, station in soil_stations.iterrows():
        station_id = station['station_id']
        station_name = station.get('name', f'Station {station_id}')

        print(f"\n[{idx+1}/{len(soil_stations)}] {station_name} (ID: {station_id})")

        try:
            start_date = binary_search_start_date(
                collector,
                station_id,
                earliest_possible,
                latest_known_good,
                parameter='HS_CV_AVG_-0.2m'
            )

            if start_date:
                print(f"  ✓ Earliest date: {start_date.date()}")
                results.append({
                    'station_id': station_id,
                    'station_name': station_name,
                    'earliest_date': start_date,
                    'days_available': (latest_known_good - start_date).days
                })
            else:
                print(f"  ✗ No data found")
                results.append({
                    'station_id': station_id,
                    'station_name': station_name,
                    'earliest_date': None,
                    'days_available': 0
                })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Create results dataframe
    results_df = pd.DataFrame(results)

    if not results_df.empty:
        # Sort by earliest date
        results_df = results_df.sort_values('earliest_date')

        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(results_df.to_string(index=False))

        # Save to CSV
        output_file = 'soil_moisture_start_dates.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")

        # Print statistics
        valid_dates = results_df[results_df['earliest_date'].notna()]
        if len(valid_dates) > 0:
            print("\nStatistics:")
            print(f"  Earliest start date: {valid_dates['earliest_date'].min().date()}")
            print(f"  Latest start date: {valid_dates['earliest_date'].max().date()}")
            print(f"  Stations with data: {len(valid_dates)}/{len(results_df)}")
            print(f"  Average days available: {valid_dates['days_available'].mean():.0f}")

    return results_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Find when stations started having soil moisture data')
    parser.add_argument('--station', type=int, help='Test single station ID')
    args = parser.parse_args()

    find_all_start_dates(test_single_station=args.station)
