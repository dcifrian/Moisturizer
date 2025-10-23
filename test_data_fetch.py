"""
Test actual historical data fetching from MeteoGalicia API
This tests the data collection that was failing
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time

BASE_URL = "https://servizos.meteogalicia.gal/mgrss/observacion/datosDiariosEstacionsMeteo.action"
STATIONS_URL = "https://servizos.meteogalicia.gal/mgrss/observacion/listaEstacionsMeteo.action"

print("=" * 60)
print("Testing Historical Data Fetching")
print("=" * 60)

# Get a couple of stations to test with
print("\n1. Fetching station list...")
response = requests.get(STATIONS_URL, timeout=30)
stations_data = response.json()
test_stations = [stations_data['listaEstacionsMeteo'][i]['idEstacion'] for i in range(2)]
print(f"Using test stations: {test_stations}")

# Test fetching last 7 days of data
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

print(f"\n2. Fetching data from {start_date.strftime('%d/%m/%Y')} to {end_date.strftime('%d/%m/%Y')}")

# Parameters to collect
parameters = [
    'HS_CV_AVG_-0.2m',  # Soil moisture
    'PP_SUM_1.5m',      # Precipitation
    'TA_AVG_1.5m',      # Temperature
    'HR_AVG_1.5m',      # Humidity
]

params = {
    'idEst': ','.join(map(str, test_stations)),
    'idParam': ','.join(parameters),
    'dataIni': start_date.strftime('%d/%m/%Y'),
    'dataFin': end_date.strftime('%d/%m/%Y')
}

print(f"\nRequest parameters:")
print(f"  Stations: {params['idEst']}")
print(f"  Parameters: {params['idParam']}")
print(f"  Date range: {params['dataIni']} to {params['dataFin']}")

try:
    print("\n3. Making API request...")
    response = requests.get(BASE_URL, params=params, timeout=30)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"\n4. Parsing response...")
        print(f"Response keys: {list(data.keys())}")

        daily_data_list = data.get('listDatosDiarios', [])
        print(f"Number of daily records: {len(daily_data_list)}")

        if len(daily_data_list) > 0:
            print(f"\n5. Examining first daily record...")
            first_record = daily_data_list[0]
            print(f"Date: {first_record.get('data')}")
            print(f"Number of stations in record: {len(first_record.get('listaEstacions', []))}")

            if len(first_record.get('listaEstacions', [])) > 0:
                first_station = first_record['listaEstacions'][0]
                print(f"Station ID: {first_station.get('idEstacion')}")
                print(f"Number of measurements: {len(first_station.get('listaMedidas', []))}")

                if len(first_station.get('listaMedidas', [])) > 0:
                    print(f"\nFirst few measurements:")
                    for measure in first_station['listaMedidas'][:3]:
                        print(f"  - {measure.get('codigoParametro')}: {measure.get('valor')} {measure.get('unidade')}")

            # Now let's parse it into a DataFrame like Moisturizer.py does
            print(f"\n6. Parsing into DataFrame...")
            rows = []
            for daily_data in data['listDatosDiarios']:
                date = daily_data.get('data')
                for station in daily_data.get('listaEstacions', []):
                    station_id = station.get('idEstacion')
                    for measure in station.get('listaMedidas', []):
                        rows.append({
                            'date': date,
                            'station_id': station_id,
                            'parameter_code': measure.get('codigoParametro'),
                            'value': measure.get('valor'),
                            'unit': measure.get('unidade'),
                            'validation_code': measure.get('lnCodigoValidacion')
                        })

            df = pd.DataFrame(rows)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')

                print(f"✓ Created DataFrame with {len(df)} rows")
                print(f"\nDataFrame info:")
                print(df.info())
                print(f"\nFirst few rows:")
                print(df.head(10))
                print(f"\nUnique parameters found:")
                print(df['parameter_code'].unique())
            else:
                print("✗ DataFrame is empty after parsing!")
        else:
            print("✗ listDatosDiarios is empty!")
            print(f"\nFull response:")
            print(json.dumps(data, indent=2))
    else:
        print(f"✗ Request failed with status {response.status_code}")
        print(f"Response: {response.text[:500]}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test complete")
